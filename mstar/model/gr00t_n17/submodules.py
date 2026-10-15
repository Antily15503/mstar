"""NodeSubmodule wrappers for the GR00T N1.7 model nodes.

Each submodule is the adapter between M*'s graph (named tensors routed along
edges) and a plain PyTorch component in ``components/``. The components know
nothing about M*; the submodules know nothing about the maths.

  Gr00tStateEncoderSubmodule -- embodiment-conditioned proprioception encoder.
                                Stateless, runs once during ``prefill``.
  Gr00tDiTSubmodule          -- one Euler step of the flow-matching action head.
                                Runs once per ``denoise`` loop iteration.
  Gr00tVLMSubmodule          -- wrapped Qwen3-VL backbone + the VL bridge.
                                Stateless one-shot, runs once during ``prefill``.
"""

import logging
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig
from mstar.model.gr00t_n17.components.dit import AlternateVLDiT
from mstar.model.gr00t_n17.components.embodiment_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from mstar.model.gr00t_n17.config import GR00TConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)


class Gr00tStateEncoderSubmodule(NodeSubmodule):
    """Encodes the robot's proprioceptive state into conditioning tokens.

    Mirrors the state half of ``Gr00tN1d7ActionHead._encode_features``::

        state = state.view(B, 1, -1)                     # flatten history
        state_features = self.state_encoder(state, embodiment_id)

    Per request it receives ``state_inputs`` of shape
    ``(state_history_length, max_state_dim)`` and an integer ``embodiment_id``,
    and emits ``state_features`` of shape ``(1, input_embedding_dim)``.

    Stateless: no KV cache, no attention, no positions -- so it declares no
    resources and needs no ``declare_step``. It runs once in ``prefill``; the
    denoise loop then re-reads its output as a persisted tensor.
    """

    def __init__(self, state_encoder: CategorySpecificMLP, config: GR00TConfig):
        super().__init__()
        self.state_encoder = state_encoder
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        state = inputs["state_inputs"][0]
        if state.dim() == 1:
            # A bare state vector -- treat it as a single history step.
            state = state.unsqueeze(0)
        expected_t = self.config.state_history_length
        if state.shape[0] != expected_t:
            raise ValueError(
                f"state_inputs has {state.shape[0]} history steps, expected "
                f"{expected_t} (config.state_history_length)"
            )
        # Flatten history into the feature dim: [T, state_dim] -> [T * state_dim]
        flat_state = state.reshape(-1)

        # embodiment_id selects which weight bank the MLP uses. Absent (single
        # embodiment deployments / structural tests) it defaults to 0.
        emb = inputs.get("embodiment_id")
        embodiment_id = emb[0].reshape(()) if emb else torch.tensor(0, dtype=torch.long)
        return NodeInputs(
            tensor_inputs={"state": flat_state, "embodiment_id": embodiment_id}
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # Collate across requests: [B, 1, T * state_dim] and [B].
        state = torch.stack([i.tensor_inputs["state"] for i in inputs], dim=0).unsqueeze(1)
        embodiment_id = torch.stack(
            [i.tensor_inputs["embodiment_id"] for i in inputs], dim=0
        )
        return {"state": state, "embodiment_id": embodiment_id}

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs):
        return _per_request(self.forward(graph_walk, engine_inputs, **kwargs), engine_inputs.request_ids)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        **kwargs,  # coming from preprocess output
    ) -> NameToTensorList:
        # state: [B, 1, T * state_dim] -> state_features: [B, 1, input_embedding_dim]
        weight_dtype = self.state_encoder.layer1.W.dtype
        features = self.state_encoder(state.to(weight_dtype), embodiment_id.long())
        # One [1, input_embedding_dim] tensor per request. embodiment_id is
        # echoed back unchanged so it can be persisted for the denoise walk --
        # the DiT's action encoder/decoder need it too, and a walk's own inputs
        # are not visible to the next walk.
        return {
            "state_features": [features[i] for i in range(features.shape[0])],
            # Sliced, not indexed: (1,) rather than 0-d, because the conductor
            # shards every graph-edge tensor on dim 0.
            "embodiment_id": [embodiment_id[i : i + 1] for i in range(embodiment_id.shape[0])],
        }


class Gr00tDiTSubmodule(NodeSubmodule):
    """One Euler step of GR00T's flow-matching action head.

    GR00T runs its denoise as a Python ``for`` loop inside
    ``Gr00tN1d7ActionHead.get_action_with_features``. In mstar the loop is the
    graph's ``Loop`` section, so this submodule implements exactly ONE
    iteration of it::

        af      = action_encoder(actions, t_discretized, embodiment_id)
        af     += position_embedding(arange(T))            # add_pos_embed
        sa_embs = cat([state_features, af], dim=1)
        out     = dit(sa_embs, encoder_hidden_states=vl_embeds, timestep=...,
                      image_mask=..., backbone_attention_mask=...)
        pred    = action_decoder(out, embodiment_id)
        actions = actions + dt * pred[:, -action_horizon:]  # Euler

    Inputs split into two kinds:
      * **evolving**, looped back by name -- ``noisy_actions`` (the chunk being
        refined) and ``timestep_index`` (the step counter).
      * **frozen conditioning**, external loop inputs that arrive once from
        ``persist_signals`` and are re-presented each iteration --
        ``state_features``, ``vl_embeds``, ``image_mask``,
        ``backbone_attention_mask``, ``embodiment_id``.

    On the first iteration ``noisy_actions``/``timestep_index`` arrive empty:
    the initial Gaussian draw and ``t=0`` are created here, mirroring how pi05
    seeds its action_gen loop inside the submodule rather than the graph.

    Stateless with respect to mstar resources: cross-attention context is
    passed as a plain tensor, so there is no KV cache and no ``declare_step``.
    """

    # Conditioning tensors that are re-presented unchanged on every iteration.
    FROZEN_INPUTS = (
        "state_features",
        "vl_embeds",
        "image_mask",
        "backbone_attention_mask",
        "embodiment_id",
    )

    def __init__(
        self,
        dit: AlternateVLDiT,
        action_encoder: MultiEmbodimentActionEncoder,
        action_decoder: CategorySpecificMLP,
        position_embedding: torch.nn.Embedding | None,
        config: GR00TConfig,
    ):
        super().__init__()
        self.dit = dit
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.position_embedding = position_embedding
        self.config = config

    @property
    def dt(self) -> float:
        """Euler step size: the schedule is traversed in N equal steps."""
        return 1.0 / self.config.num_inference_timesteps

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        def _first(name: str) -> torch.Tensor | None:
            vals = inputs.get(name)
            return vals[0] if vals else None

        actions = _first("noisy_actions")
        if actions is None:
            # First iteration: one Gaussian draw for the whole chunk. Every
            # later iteration refines this same tensor (Euler), so the noise is
            # sampled exactly once per request. Seeded from the conductor's
            # per-request seed (as pi05 does) so a served request is
            # reproducible, and drawn on the model's device.
            device = self.get_device()
            generator = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
            actions = torch.randn(
                self.config.action_horizon, self.config.max_action_dim,
                device=device, generator=generator,
            )

        step = _first("timestep_index")
        step = torch.zeros(1, dtype=torch.long) if step is None else step.reshape(1)

        tensors: dict[str, torch.Tensor] = {"noisy_actions": actions, "timestep_index": step}
        for name in self.FROZEN_INPUTS:
            value = _first(name)
            if value is None:
                raise ValueError(
                    f"DiT node is missing required conditioning input {name!r}; "
                    "it should arrive from the prefill walk's persist signals"
                )
            tensors[name] = value
        # Checked per request here (the engine fails just this request), not in
        # preprocess, where a raise would fail the whole batch.
        if tensors["vl_embeds"].shape[0] > self.config.cuda_graph_context_len:
            raise ValueError(
                f"VLM context of {tensors['vl_embeds'].shape[0]} tokens exceeds "
                f"cuda_graph_context_len={self.config.cuda_graph_context_len}"
            )
        return NodeInputs(tensor_inputs=tensors)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # The VLM context varies per request (prompt length); pad it to one fixed
        # length so every batch has the shapes the CUDA graphs captured (and
        # compiled code never recompiles on a new prompt length). The pad is
        # masked out of backbone_attention_mask, which every cross-attention
        # honours, so it changes no row beyond kernel noise (~2 bf16 ulps).
        n, device = self.config.cuda_graph_context_len, self.get_device()

        def _stack(name: str) -> torch.Tensor:
            # First-step inputs are built here, later ones come off the engine on
            # device; a batch can mix both.
            ts = [i.tensor_inputs[name].to(device) for i in inputs]
            if name in ("vl_embeds", "image_mask", "backbone_attention_mask"):
                ts = [torch.cat([t, t.new_zeros((n - t.shape[0], *t.shape[1:]))]) for t in ts]
            return torch.stack(ts, dim=0)

        return {
            "noisy_actions": _stack("noisy_actions"),                   # [B, H, action_dim]
            "timestep_index": _stack("timestep_index"),                 # [B, 1]
            "state_features": _stack("state_features"),                 # [B, 1, emb]
            "vl_embeds": _stack("vl_embeds"),                           # [B, S, backbone_dim]
            "image_mask": _stack("image_mask"),                         # [B, S]
            "backbone_attention_mask": _stack("backbone_attention_mask"),  # [B, S]
            "embodiment_id": _stack("embodiment_id"),                   # [B, 1]
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        noisy_actions: torch.Tensor,
        timestep_index: torch.Tensor,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        embodiment_id: torch.Tensor,
        **kwargs,  # coming from preprocess output
    ) -> NameToTensorList:
        batch = noisy_actions.shape[0]
        weight = next(self.dit.parameters())
        dtype, device = weight.dtype, weight.device
        actions = noisy_actions.to(device=device, dtype=dtype)
        embodiment_id = embodiment_id.reshape(-1).long().to(device)  # [B, 1] off the edge -> [B]

        # Per request: nothing keeps batched requests on the same loop iteration.
        # Device-side only (no .item()), so the step is CUDA-graph capturable.
        step = timestep_index.reshape(-1).to(device)
        timesteps = (
            step.double() / self.config.num_inference_timesteps * self.config.num_timestep_buckets
        ).long()

        action_features = self.action_encoder(actions, timesteps, embodiment_id)
        if self.position_embedding is not None:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=action_features.device
            )
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        # State tokens lead, action tokens follow -- the decoder slices the
        # action tail back off below, so this order is load-bearing.
        sa_embs = torch.cat([state_features.to(device=device, dtype=dtype), action_features], dim=1)

        model_output = self.dit(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds.to(device=device, dtype=dtype),
            timestep=timesteps,
            image_mask=image_mask.to(device).bool(),
            backbone_attention_mask=backbone_attention_mask.to(device).bool(),
        )
        pred = self.action_decoder(model_output, embodiment_id)

        # The DiT sees state+action tokens; only the action tail is a velocity.
        pred_velocity = pred[:, -self.config.action_horizon :]
        actions = actions + self.dt * pred_velocity

        next_step = step + 1
        return {
            "noisy_actions": [actions[i] for i in range(batch)],
            "timestep_index": [next_step[i : i + 1] for i in range(batch)],  # (1,): crosses an edge
        }

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs):
        return _per_request(self.forward(graph_walk, engine_inputs, **kwargs), engine_inputs.request_ids)

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list:
        """One denoise step, captured per batch size at the fixed context length.

        The dummy row (batch padding) keeps one image token and every context
        token live, so no cross-attention row is fully masked.
        """
        n, cfg = self.config.cuda_graph_context_len, self.config
        dtype = next(self.dit.parameters()).dtype
        row = NodeInputs(tensor_inputs={
            "noisy_actions": torch.zeros(cfg.action_horizon, cfg.max_action_dim, device=device),
            "timestep_index": torch.zeros(1, dtype=torch.long, device=device),
            "state_features": torch.zeros(1, cfg.input_embedding_dim, device=device, dtype=dtype),
            "vl_embeds": torch.zeros(n, cfg.backbone_embedding_dim, device=device, dtype=dtype),
            "image_mask": torch.arange(n, device=device) == 0,
            "backbone_attention_mask": torch.ones(n, dtype=torch.bool, device=device),
            "embodiment_id": torch.zeros(1, dtype=torch.long, device=device),
        })
        return [BatchedCudaGraphConfig(
            capture_graph_walk="denoise",
            single_request_inputs=row,
            capture_batch_sizes=cfg.cuda_graph_batch_sizes,
        )]


class Gr00tVLMSubmodule(NodeSubmodule):
    """Cosmos-Reason2-2B (Qwen3-VL) backbone + the VL bridge.

    Wraps the HF model rather than hand-porting it: GR00T calls the backbone
    exactly once per request (``backbone_outputs = self.backbone(...)``), so it
    is a stateless one-shot encoder with no KV reuse -- the same situation as
    whisper's encoder and pi05's SigLIP, both of which mstar wraps.

    It also owns the **VL bridge** (``vlln`` + ``vl_self_attention``). GR00T
    keeps those on the action head, but calls them once in
    ``process_backbone_output`` before the denoise loop, so in mstar they belong
    here -- running them in the loop body would repeat them once per Euler step.
    NOTE for weight loading: their checkpoint keys therefore sit under the
    action-head prefix while this submodule owns the modules.

    Emits the three tensors the DiT needs::

        vl_embeds               [B, S, backbone_embedding_dim]  (bridged)
        image_mask              [B, S]  input_ids == image_token_id
        backbone_attention_mask [B, S]  attention_mask == 1

    Inputs are the Qwen3-VL processor's outputs, matching
    ``Qwen3Backbone.forward``'s ``keys_to_use``.
    """

    HF_INPUTS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")

    def __init__(self, backbone, vlln, vl_self_attention, image_token_id: int, config: GR00TConfig):
        super().__init__()
        self.backbone = backbone
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention
        self.image_token_id = image_token_id
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        tensors = {}
        for name in self.HF_INPUTS:
            vals = inputs.get(name)
            if vals:
                tensors[name] = vals[0]
        if "input_ids" not in tensors:
            raise ValueError("vlm node requires input_ids from the Qwen3-VL processor")
        return NodeInputs(tensor_inputs=tensors)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # Sequential path is bs=1. Batching would need equal token counts (or
        # padding) across requests before this can stack; can_batch must gate it.
        # Qwen3-VL's batch layout differs per key: token tensors get a batch
        # dim, while pixel_values / image_grid_thw are flat across every image
        # of every request (patches x dim, images x 3) -- concatenated, never
        # stacked.
        out: dict[str, torch.Tensor | Any] = {}
        for name in self.HF_INPUTS:
            present = [i.tensor_inputs[name] for i in inputs if name in i.tensor_inputs]
            if not present:
                continue
            if name in ("pixel_values", "image_grid_thw"):
                out[name] = torch.cat(present, dim=0)
            else:
                out[name] = torch.stack(present, dim=0)
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs,  # coming from preprocess output
    ) -> NameToTensorList:
        vl_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        if pixel_values is not None:
            vl_input["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            vl_input["image_grid_thw"] = image_grid_thw
            # transformers >= 5 computes Qwen3-VL's 3D RoPE positions from
            # mm_token_type_ids (0 text, 1 image, 2 video) and refuses to run
            # without it; 4.57 (what the reference pins) derived them from
            # input_ids. Same information, so derive it here rather than add
            # a fifth graph input: image tokens are exactly image_token_id.
            vl_input["mm_token_type_ids"] = (input_ids == self.image_token_id).long()

        outputs = self.backbone(**vl_input, output_hidden_states=True)
        # The LLM stack was truncated to ``select_layer`` at construction, so
        # the last hidden state IS the selected layer (this is how GR00T does
        # it -- popping layers rather than indexing, which also saves compute).
        features = outputs.hidden_states[-1]

        # The bridge: LayerNorm then self-attention, applied ONCE here rather
        # than per denoise step.
        features = self.vl_self_attention(self.vlln(features))

        image_mask = input_ids == self.image_token_id
        if attention_mask is None:
            backbone_attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            backbone_attention_mask = attention_mask == 1

        batch = features.shape[0]
        return {
            "vl_embeds": [features[i] for i in range(batch)],
            "image_mask": [image_mask[i] for i in range(batch)],
            "backbone_attention_mask": [backbone_attention_mask[i] for i in range(batch)],
        }


def _per_request(outputs: NameToTensorList, request_ids) -> dict:
    """``forward``'s per-request lists (batch order) -> ``forward_batched``'s {rid: outputs}."""
    return {rid: {name: [ts[i]] for name, ts in outputs.items()} for i, rid in enumerate(request_ids)}
