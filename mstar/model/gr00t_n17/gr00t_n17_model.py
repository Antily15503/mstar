"""GR00T N1.7 Model subclass — declares the compute graph, walks, and the
per-request walk state machine for M*.

GR00T N1.7 is a vision-language-action policy: image(s) + a language
instruction + the robot's proprioceptive state in, a continuous action
trajectory out. Three nodes, faithful to ``Gr00tN1d7ActionHead``:

    vlm            Cosmos-Reason2-2B (Qwen3-VL). The vision tower is INSIDE it,
                   so there is no standalone image encoder (contrast pi05,
                   which splits its SigLIP out). Also owns the VL bridge.
    state_encoder  CategorySpecificMLP, embodiment-indexed.
    dit            AlternateVLDiT flow-matching action head.

    prefill = Parallel([vlm, state_encoder])    # conditioning, computed once
    denoise = Loop(dit, max_iters=4) -> emit action

The DiT **cross-attends** the VLM output (``encoder_hidden_states=vl_embeds``);
it does not share a KV cache the way pi05's action expert does. Conditioning
therefore crosses the walk boundary as *persist signals* rather than living in
a resource -- GR00T consumes those tensors on every denoise step.

Status: all three nodes are implemented; the DiT is parity-tested bit-exact
against the reference. The ``vlm`` node stays in dummy mode until an authorized
HF token is available -- nvidia/Cosmos-Reason2-2B is a GATED repo. Weight
loading for the other nodes is Phase 4 (they are random-init today).
"""

import json
import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import NodeResourceSpec
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.gr00t_n17.components.dit import AlternateVLDiT, SelfAttentionTransformer
from mstar.model.gr00t_n17.components.embodiment_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from mstar.model.gr00t_n17.config import GR00TConfig, load_gr00t_config
from mstar.model.gr00t_n17.submodules import (
    Gr00tDiTSubmodule,
    Gr00tStateEncoderSubmodule,
    Gr00tVLMSubmodule,
)
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


class Gr00tN17Model(Model):
    """GR00T N1.7 vision-language-action model."""

    PREFILL_WALK = "prefill"
    DENOISE_WALK = "denoise"

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        **kwargs
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        # Yaml-driven GR00TConfig overrides forwarded by the entrypoint (any
        # key under `model_kwargs:` in configs/gr00t_n17.yaml). Applied inside
        # _load_config() *last* — after the checkpoint's config.json — so yaml
        # wins over both the checkpoint and the dataclass defaults, and weight
        # shapes are materialized from consistent values.
        self._yaml_config_overrides = dict(kwargs)
        self.config: GR00TConfig = self._load_config()
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

        # Components, materialized lazily by get_submodule() — GR00T's 3 nodes.
        # Node count mirrors GR00T's own architecture: the vision tower lives
        # INSIDE the Cosmos-Reason2-2B (Qwen3-VL) backbone, so there is no
        # standalone image encoder to split out (contrast pi05, which does
        # separate its SigLIP `vit_encoder` from the LLM).
        # The vlm node owns the backbone plus the VL bridge (vlln +
        # vl_self_attention), which GR00T keeps on the action head but runs
        # once, before the denoise loop.
        self.vlm = None             # Cosmos-Reason2-2B backbone (vision + language)
        self.vlln = None
        self.vl_self_attention = None
        self._image_token_id: int | None = None
        self.state_encoder = None   # CategorySpecificMLP, embodiment-indexed
        # The dit node owns four modules: the transformer plus the action
        # encoder/decoder that bracket it and the action position embedding.
        self.dit = None             # AlternateVLDiT flow-matching action head
        self.action_encoder = None  # MultiEmbodimentActionEncoder
        self.action_decoder = None  # CategorySpecificMLP -> velocity
        self.position_embedding = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self) -> GR00TConfig:
        if self.skip_weight_loading:
            return load_gr00t_config(None)  # defaults only
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=self.model_path_hf, filename="config.json", cache_dir=self.cache_dir
            )
            with open(config_path) as f:
                return load_gr00t_config(json.load(f))
        except Exception as exc:
            logger.warning("Could not load GR00T config from HF (%s); using defaults.", exc)
            return load_gr00t_config(None)

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """GR00T declares no engine resources -- and that is the correct answer,
        not a stub.

        Resources exist for state the engine must manage across forward passes:
        a paged KV cache, its attention plan, rotary positions, a sampler. None
        of GR00T's three nodes need any:

        * ``vlm`` -- called exactly ONCE per request, no token generation, so
          nothing is cached between passes. (A resource-free node is normal:
          pi05's ``vit_encoder`` appears in no spec either.)
        * ``state_encoder`` -- a two-layer MLP.
        * ``dit`` -- iterates, but its conditioning is a persisted tensor and
          its attention is diffusers' own, not mstar's paged attention. Nothing
          grows across the four Euler steps.

        No ``SamplerSpec``: actions are continuous, there is no vocabulary.

        The cost is that mstar's paged-attention and CUDA-graph fast paths do
        not apply to the DiT loop. Revisiting that -- a ``CrossAttentionSpec``
        to cache the projected cross-attention K/V once instead of
        re-projecting it on every step -- is a Phase 5 performance question,
        not a correctness one. See 06 - Gaps and Improvement Ideas.
        """
        return []

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # GR00T splits cleanly into "encode the conditioning once" then "denoise
        # the action chunk N times", mirroring Gr00tN1d7ActionHead.get_action():
        # _encode_features() runs once, get_action_with_features() loops.
        #
        # prefill — the two conditioning branches are independent, so they run
        # in Parallel:
        #   vlm           : image + text -> vl_embeds  [B, seq, 2048]
        #   state_encoder : proprio state -> state_features  [B, 1, 1536]
        # Both leave the walk as *persist signals* (EMPTY_DESTINATION +
        # persist=True) rather than resource writes, because GR00T consumes them
        # as live tensors inside every denoise step, not as a frozen KV prefix:
        # state_features is concatenated into hidden_states and vl_embeds is the
        # cross-attention context. (This is why pi05's `outputs=[]` prefill does
        # NOT transfer here — pi05 hides its conditioning in a shared KV cache.)
        prefill = Parallel(
            [
                GraphNode(
                    name="vlm",
                    # The Qwen3-VL processor's outputs, matching
                    # Qwen3Backbone.forward's keys_to_use.
                    input_names=list(Gr00tVLMSubmodule.HF_INPUTS),
                    outputs=[
                        # The DiT needs more than the features: AlternateVLDiT
                        # splits its cross-attention between the VLM's text and
                        # image tokens, so it needs to know which positions are
                        # images (image_mask) and which are padding
                        # (backbone_attention_mask).
                        GraphEdge(
                            next_node=EMPTY_DESTINATION, name="vl_embeds", persist=True
                        ),
                        GraphEdge(
                            next_node=EMPTY_DESTINATION, name="image_mask", persist=True
                        ),
                        GraphEdge(
                            next_node=EMPTY_DESTINATION,
                            name="backbone_attention_mask",
                            persist=True,
                        ),
                    ],
                ),
                GraphNode(
                    name="state_encoder",
                    # embodiment_id picks which per-robot weight bank the
                    # CategorySpecificMLP uses, so it is routed as its own
                    # named input rather than packed into the state vector
                    # (it is an integer index, not proprioception).
                    input_names=["state_inputs", "embodiment_id"],
                    outputs=[
                        GraphEdge(
                            next_node=EMPTY_DESTINATION, name="state_features", persist=True
                        ),
                        # Passed straight through so the DiT can reach it: its
                        # action encoder/decoder are embodiment-conditioned too,
                        # and a walk's inputs are not visible to the next walk.
                        # See 06 - Gaps: mstar has no slot for a per-request
                        # conditioning scalar, so it rides a tensor edge.
                        GraphEdge(
                            next_node=EMPTY_DESTINATION, name="embodiment_id", persist=True
                        ),
                    ],
                ),
            ]
        )

        # denoise — one Loop iteration == one Euler step of GR00T's flow
        # matching (num_inference_timesteps=4, dt=1/4):
        #   action_encoder(actions, t, embodiment) [+ pos_embed]
        #   -> cat(state_features, action_features) as hidden_states
        #   -> AlternateVLDiT(cross-attends vl_embeds) -> action_decoder
        #   -> velocity -> actions += dt * velocity
        # Only the *evolving* tensors loop back by name (noisy_actions carries
        # the Euler-updated chunk, timestep_index the step counter).
        # state_features / vl_embeds are EXTERNAL loop inputs: no loop-back edge
        # produces them, they enter once at the walk transition from
        # persist_signals and the loop re-presents the same worker-local tensor
        # each iteration (no per-step transfer). Same idiom as cosmos3's
        # cond_latents.
        # The terminal output name must match the loop-back edge name — see
        # Loop._replace_outputs_for_final_iter in mstar/graph/base.py.
        denoise = Loop(
            section=GraphNode(
                name="dit",
                input_names=[
                    # evolving: looped back by name each iteration
                    "noisy_actions",
                    "timestep_index",
                    # frozen conditioning: external inputs, entering once from
                    # persist_signals and re-presented every iteration
                    "state_features",
                    "vl_embeds",
                    "image_mask",
                    "backbone_attention_mask",
                    "embodiment_id",
                ],
                outputs=[
                    GraphEdge(next_node="dit", name="noisy_actions"),
                    GraphEdge(next_node="dit", name="timestep_index"),
                ],
            ),
            max_iters=self.config.num_inference_timesteps,
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="noisy_actions",
                    output_modality="action",
                    persist=True,
                ),
            ],
        )

        return {
            self.PREFILL_WALK: prefill,
            self.DENOISE_WALK: denoise,
        }

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        **kwargs,
    ) -> NameToTensorList:
        # TODO(phase4): run the real Qwen3-VL processor here to turn
        # prompt + images into input_ids / attention_mask / pixel_values /
        # image_grid_thw. That processor ships with the GATED
        # nvidia/Cosmos-Reason2-2B repo, so until access lands the caller
        # supplies them directly (which is also what a parity harness does).
        out: NameToTensorList = {}
        for name in Gr00tVLMSubmodule.HF_INPUTS:
            value = kwargs.get(name)
            if value is not None:
                out[name] = value if isinstance(value, list) else [value]
        embodiment_id = kwargs.get("embodiment_id")
        if embodiment_id is not None:
            out["embodiment_id"] = [
                torch.as_tensor(embodiment_id, dtype=torch.long).reshape(())
            ]
        state = kwargs.get("state_inputs", kwargs.get("robot_state"))
        if state is not None:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            out["state_inputs"] = [state]
        return out

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality == "action":
            return output.detach().to(torch.float32).cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for GR00T N1.7: {modality!r}")

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration (the walk state machine)
    # ------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=self.PREFILL_WALK,
            is_prefill=True,
            kwargs={},
        )

        # Route each external input to the node that consumes it. All four
        # processor tensors go to the VLM: the vision tower is inside the
        # Qwen3-VL backbone, so there is no separate encoder node for images.
        routing = {name: "vlm" for name in Gr00tVLMSubmodule.HF_INPUTS}
        routing["state_inputs"] = "state_encoder"
        routing["embodiment_id"] = "state_encoder"
        inputs = []
        for signal_name, node_name in routing.items():
            if signal_name in input_signals:
                edge = GraphEdge(next_node=node_name, name=signal_name)
                edge.tensor_info = input_signals[signal_name]
                inputs.append(edge)

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False
        inputs: list[GraphEdge] = []

        if metadata.graph_walk == self.PREFILL_WALK:
            # prefill done -> denoise. noisy_actions (the initial Gaussian draw)
            # and timestep_index=0 are created inside the DiT submodule's
            # preprocess, so they enter as empty edges.
            metadata.is_prefill = False
            metadata.graph_walk = self.DENOISE_WALK
            inputs = [
                GraphEdge(next_node="dit", name="noisy_actions"),
                GraphEdge(next_node="dit", name="timestep_index"),
            ]
            # The prefill walk's conditioning, persisted at the conductor, seeds
            # the loop's external inputs. All are re-presented every iteration.
            for name in Gr00tDiTSubmodule.FROZEN_INPUTS:
                edge = GraphEdge(next_node="dit", name=name)
                edge.tensor_info = persist_signals.get(name, [])
                inputs.append(edge)
        elif metadata.graph_walk == self.DENOISE_WALK:
            request_done = True

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": metadata.is_prefill},
            request_done=request_done,
        )

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> torch.nn.Module | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Loaded GR00T N1.7 submodule for %s", node_name)
        return submodule

    def _create_submodule(self, node_name: str, device: str) -> NodeSubmodule | None:
        # TODO(phase3): "vlm" (wrap HF Qwen3VLForConditionalGeneration) and
        # "dit" (AlternateVLDiT). Returning None for those keeps them in dummy
        # mode so the rest of the graph can be exercised meanwhile.
        if node_name == "state_encoder":
            self._init_state_encoder(device)
            return Gr00tStateEncoderSubmodule(
                state_encoder=self.state_encoder, config=self.config
            )
        if node_name == "vlm":
            if not self._init_vlm(device):
                return None
            return Gr00tVLMSubmodule(
                backbone=self.vlm,
                vlln=self.vlln,
                vl_self_attention=self.vl_self_attention,
                image_token_id=self._image_token_id,
                config=self.config,
            )
        if node_name == "dit":
            self._init_dit(device)
            return Gr00tDiTSubmodule(
                dit=self.dit,
                action_encoder=self.action_encoder,
                action_decoder=self.action_decoder,
                position_embedding=self.position_embedding,
                config=self.config,
            )
        return None

    def _init_state_encoder(self, device: str):
        """Build the proprioception encoder, matching Gr00tN1d7ActionHead."""
        if self.state_encoder is not None:
            return
        self.state_encoder = CategorySpecificMLP(
            num_categories=self.config.max_num_embodiments,
            input_dim=self.config.max_state_dim * self.config.state_history_length,
            hidden_dim=self.config.dit_hidden_size,
            output_dim=self.config.input_embedding_dim,
        ).to(device)
        # TODO(phase4): load real weights; this is random-init until then.

    def _init_vlm(self, device: str) -> bool:
        """Load the Qwen3-VL backbone and build the VL bridge.

        Returns False (leaving the node in dummy mode) when the backbone cannot
        be loaded -- nvidia/Cosmos-Reason2-2B is a GATED repo, so this is the
        expected path until an authorized HF token is available. Everything
        else in the model is exercisable meanwhile.
        """
        if self.vlm is not None:
            return True
        if self.skip_weight_loading:
            return False
        cfg = self.config
        try:
            from transformers import Qwen3VLForConditionalGeneration

            backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                cfg.backbone_model_name, cache_dir=self.cache_dir
            ).eval()
        except Exception as exc:
            logger.warning(
                "Could not load the GR00T backbone %s (%s). The vlm node stays "
                "in dummy mode; note the repo is gated and needs an authorized "
                "HF token.",
                cfg.backbone_model_name,
                exc,
            )
            return False

        # select_layer is applied by TRUNCATING the stack, not by indexing into
        # it: GR00T pops layers above select_layer so the last hidden state IS
        # the selected one, and the layers above are never computed.
        while len(backbone.language_model.layers) > cfg.select_layer:
            backbone.language_model.layers.pop(-1)

        self.vlm = backbone.to(device)
        self._image_token_id = backbone.config.image_token_id
        # The VL bridge lives here rather than in the denoise loop -- see the
        # Gr00tVLMSubmodule docstring.
        self.vlln = (
            torch.nn.LayerNorm(cfg.backbone_embedding_dim) if cfg.use_vlln else torch.nn.Identity()
        ).to(device)
        self.vl_self_attention = (
            SelfAttentionTransformer(
                num_attention_heads=cfg.bridge_num_heads,
                attention_head_dim=cfg.bridge_head_dim,
                num_layers=cfg.bridge_num_layers,
                max_num_positional_embeddings=cfg.max_seq_len,
            ).to(device)
            if cfg.bridge_num_layers > 0
            else torch.nn.Identity()
        )
        # TODO(phase4): load the bridge's real weights. Their checkpoint keys
        # sit under the ACTION HEAD prefix (vlln.*, vl_self_attention.*) even
        # though this node owns the modules.
        return True

    def _init_dit(self, device: str):
        """Build the flow-matching action head, matching Gr00tN1d7ActionHead."""
        if self.dit is not None:
            return
        cfg = self.config
        self.dit = AlternateVLDiT(
            num_attention_heads=cfg.dit_num_heads,
            attention_head_dim=cfg.dit_head_dim,
            output_dim=cfg.dit_hidden_size,
            num_layers=cfg.dit_num_layers,
            norm_type="ada_norm",
            interleave_self_attention=cfg.dit_interleave_self_attention,
            cross_attention_dim=cfg.backbone_embedding_dim,
            attend_text_every_n_blocks=cfg.attend_text_every_n_blocks,
            max_num_positional_embeddings=cfg.max_seq_len,
        ).to(device)
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=cfg.max_action_dim,
            hidden_size=cfg.input_embedding_dim,
            num_embodiments=cfg.max_num_embodiments,
        ).to(device)
        self.action_decoder = CategorySpecificMLP(
            num_categories=cfg.max_num_embodiments,
            input_dim=cfg.dit_hidden_size,
            hidden_dim=cfg.dit_hidden_size,
            output_dim=cfg.max_action_dim,
        ).to(device)
        self.position_embedding = (
            torch.nn.Embedding(cfg.max_seq_len, cfg.input_embedding_dim).to(device)
            if cfg.add_pos_embed
            else None
        )
        # TODO(phase4): load real weights; this is random-init until then.
