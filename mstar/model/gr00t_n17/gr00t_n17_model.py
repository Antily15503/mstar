"""GR00T N1.7 Model subclass — declares the compute graph, walks, and the
per-request walk state machine for M*.

TODO(notes-pass): full module docstring.

Graph shape (faithful to Gr00tN1d7ActionHead): 3 nodes — ``vlm`` (Cosmos-
Reason2-2B / Qwen3-VL, vision tower included), ``state_encoder``
(CategorySpecificMLP, embodiment-indexed) and ``dit`` (AlternateVLDiT flow-
matching head). The DiT **cross-attends** the VLM output
(``encoder_hidden_states=vl_embeds``); it does not share a KV cache the way
pi05's action expert does.

Scaffold status (Phase 2): structure + walk state machine are real; forward
compute is stubbed. ``get_submodule`` returns ``None`` (dummy mode), so M* can
*load* GR00T with no weights. ``get_node_resources`` and the neural-net
components are Phase 3 — the remaining blocker is the gated
nvidia/Cosmos-Reason2-2B repo (backbone KV dims).
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
from mstar.model.gr00t_n17.config import GR00TConfig, load_gr00t_config
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
        self.vlm = None             # Cosmos-Reason2-2B backbone (vision + language)
        self.state_encoder = None   # CategorySpecificMLP, embodiment-indexed
        self.dit = None             # AlternateVLDiT flow-matching action head

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
        # TODO(phase3): declare the VLM's KV/Attn/Pos specs and the DiT's
        # attention spec. Still blocked on the backbone's KV dims
        # (num_layers/kv_heads/head_dim), which live in the GATED
        # nvidia/Cosmos-Reason2-2B config.
        # Fusion is resolved: the DiT cross-attends vl_embeds, so its context
        # is a persisted tensor, not a shared KV prefix. Passing vl_embeds
        # straight through as a loop input is the correct-and-simple v1; a
        # CrossAttentionSpec (caching the projected K/V once instead of
        # re-projecting on all 4 steps) is a later optimization — see
        # [[06 - Gaps and Improvement Ideas]].
        # Empty is valid for the dummy-load scaffold.
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
                    input_names=["image_inputs", "text_inputs"],
                    outputs=[
                        GraphEdge(
                            next_node=EMPTY_DESTINATION, name="vl_embeds", persist=True
                        ),
                    ],
                ),
                GraphNode(
                    name="state_encoder",
                    input_names=["state_inputs"],
                    outputs=[
                        GraphEdge(
                            next_node=EMPTY_DESTINATION, name="state_features", persist=True
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
                    "noisy_actions",
                    "timestep_index",
                    "state_features",
                    "vl_embeds",
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
        # TODO(phase3): real preprocessing — Cosmos-Reason2-2B tokenization for
        # text, SigLip2 image transform, and raw float state + embodiment id for
        # the MLP encoder. This scaffold produces structurally-valid tensors so
        # the graph can be exercised without weights.
        out: NameToTensorList = {}
        if prompt is not None:
            out["text_inputs"] = [torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long)]
        image = kwargs.get("image_inputs", kwargs.get("image"))
        if image is not None:
            out["image_inputs"] = image if isinstance(image, list) else [image]
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

        # Route each external input to the node that consumes it. Images and
        # text both go to the VLM: the vision tower is inside the Qwen3-VL
        # backbone, so there is no separate encoder node to route images to.
        routing = {
            "image_inputs": "vlm",
            "text_inputs": "vlm",
            "state_inputs": "state_encoder",
        }
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
            # the loop's external inputs. Both are re-presented every iteration.
            for name in ("state_features", "vl_embeds"):
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
        # TODO(phase3): build real submodules (image_encoder / state_encoder /
        # vlm / dit) from submodules.py + components/. Returning None = dummy
        # mode: M* loads the model and builds the plan without weights.
        return self._submodule_cache.get(node_name)
