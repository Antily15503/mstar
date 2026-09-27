"""NodeSubmodule wrappers for the GR00T N1.7 model nodes.

Each submodule is the adapter between M*'s graph (named tensors routed along
edges) and a plain PyTorch component in ``components/``. The components know
nothing about M*; the submodules know nothing about the maths.

  Gr00tStateEncoderSubmodule -- embodiment-conditioned proprioception encoder.
                                Stateless, runs once during ``prefill``.
  Gr00tDiTSubmodule          -- one Euler step of the flow-matching action head.
                                Runs once per ``denoise`` loop iteration.

TODO(phase3): Gr00tVLMSubmodule, wrapping HF Qwen3VLForConditionalGeneration.
It also owns the VL bridge (``vlln`` + ``vl_self_attention``): GR00T keeps
those on the action head but calls them once, in ``process_backbone_output``,
before the denoise loop -- so in mstar they belong to the prefill node, not the
loop body. NOTE for weight loading: their checkpoint keys therefore sit under
the action-head prefix while the vlm submodule owns the modules.
"""

import logging
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
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
            "embodiment_id": [embodiment_id[i] for i in range(embodiment_id.shape[0])],
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

    def _discretize(self, step: int) -> int:
        """Continuous time t/N -> one of ``num_timestep_buckets`` buckets."""
        t_cont = step / float(self.config.num_inference_timesteps)
        return int(t_cont * self.config.num_timestep_buckets)

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
            # sampled exactly once per request.
            actions = torch.randn(self.config.action_horizon, self.config.max_action_dim)

        step = _first("timestep_index")
        step = torch.tensor(0, dtype=torch.long) if step is None else step.reshape(())

        tensors: dict[str, torch.Tensor] = {"noisy_actions": actions, "timestep_index": step}
        for name in self.FROZEN_INPUTS:
            value = _first(name)
            if value is None:
                raise ValueError(
                    f"DiT node is missing required conditioning input {name!r}; "
                    "it should arrive from the prefill walk's persist signals"
                )
            tensors[name] = value
        return NodeInputs(tensor_inputs=tensors)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # Batching stacks the context, so it requires equal VLM sequence length
        # across the batch. can_batch must enforce that before this is enabled;
        # the sequential path is bs=1.
        def _stack(name: str) -> torch.Tensor:
            return torch.stack([i.tensor_inputs[name] for i in inputs], dim=0)

        return {
            "noisy_actions": _stack("noisy_actions"),                   # [B, H, action_dim]
            "timestep_index": _stack("timestep_index"),                 # [B]
            "state_features": _stack("state_features"),                 # [B, 1, emb]
            "vl_embeds": _stack("vl_embeds"),                           # [B, S, backbone_dim]
            "image_mask": _stack("image_mask"),                         # [B, S]
            "backbone_attention_mask": _stack("backbone_attention_mask"),  # [B, S]
            "embodiment_id": _stack("embodiment_id"),                   # [B]
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
        dtype = next(self.dit.parameters()).dtype
        actions = noisy_actions.to(dtype)
        embodiment_id = embodiment_id.long()

        # Every request in a batch is on the same loop iteration, so one step
        # index drives the whole batch.
        step = int(timestep_index.reshape(-1)[0].item())
        timesteps = torch.full(
            (batch,), self._discretize(step), device=actions.device, dtype=torch.long
        )

        action_features = self.action_encoder(actions, timesteps, embodiment_id)
        if self.position_embedding is not None:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=action_features.device
            )
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        # State tokens lead, action tokens follow -- the decoder slices the
        # action tail back off below, so this order is load-bearing.
        sa_embs = torch.cat([state_features.to(dtype), action_features], dim=1)

        model_output = self.dit(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds.to(dtype),
            timestep=timesteps,
            image_mask=image_mask.bool(),
            backbone_attention_mask=backbone_attention_mask.bool(),
        )
        pred = self.action_decoder(model_output, embodiment_id)

        # The DiT sees state+action tokens; only the action tail is a velocity.
        pred_velocity = pred[:, -self.config.action_horizon :]
        actions = actions + self.dt * pred_velocity

        next_step = torch.tensor(step + 1, dtype=torch.long)
        return {
            "noisy_actions": [actions[i] for i in range(batch)],
            "timestep_index": [next_step.clone() for _ in range(batch)],
        }
