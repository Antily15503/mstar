"""NodeSubmodule wrappers for the GR00T N1.7 model nodes.

Each submodule is the adapter between M*'s graph (named tensors routed along
edges) and a plain PyTorch component in ``components/``. The components know
nothing about M*; the submodules know nothing about the maths.

  Gr00tStateEncoderSubmodule -- embodiment-conditioned proprioception encoder.
                                Stateless, runs once during ``prefill``.

TODO(phase3): Gr00tVLMSubmodule (wraps HF Qwen3VLForConditionalGeneration) and
Gr00tDiTSubmodule (AlternateVLDiT + action encoder/decoder + Euler step).
"""

import logging
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.gr00t_n17.components.embodiment_mlp import CategorySpecificMLP
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
        # One [1, input_embedding_dim] tensor per request.
        return {"state_features": [features[i] for i in range(features.shape[0])]}
