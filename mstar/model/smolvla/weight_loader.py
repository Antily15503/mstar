"""Where each tensor of a SmolVLA checkpoint goes.

``lerobot/smolvla_base`` keeps the VLM, the action expert and the small
projections around them under ``model.``. Keys route to *components*, not
nodes: which node holds which component is still open (D06). Inside a
component a tensor keeps its reference name minus the component's root. The
text layers' separate q/k/v and gate/up load into fused parameters.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mstar.model.loader.base import LLAMA_STACKED_PARAMS, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards

_VLM = "model.vlm_with_expert.vlm.model."
_EXPERT = "model.vlm_with_expert.lm_expert."

# component -> (root it strips, checkpoint prefixes it owns)
COMPONENT_PREFIXES: dict[str, tuple[str, tuple[str, ...]]] = {
    "vision": (_VLM + "vision_model.", (_VLM + "vision_model.",)),
    "connector": (_VLM + "connector.modality_projection.", (_VLM + "connector.",)),
    "text": (_VLM + "text_model.", (_VLM + "text_model.",)),
    "state": ("model.", ("model.state_proj.",)),
    "expert": (_EXPERT, (_EXPERT,)),
    "action": ("model.", (
        "model.action_in_proj.",
        "model.action_time_mlp_in.",
        "model.action_time_mlp_out.",
        "model.action_out_proj.",
    )),
}

# Tensors no SmolVLA module holds. The prefill's output is discarded
# (`modeling_smolvla.py:801`): only its K/V leave it, so the text model needs no
# final norm and no logits. Layer 15's q, o and MLP are dead for the same
# reason, but stay with their layer for now (C04 decides).
SKIPPED = frozenset({
    _VLM + "text_model.norm.weight",
    "model.vlm_with_expert.vlm.lm_head.weight",
})

COMPONENT_STACKED = {"text": LLAMA_STACKED_PARAMS}


def component_of(name: str) -> str | None:
    """The component a checkpoint tensor loads into; None for a named skip."""
    if name in SKIPPED:
        return None
    for component, (_, prefixes) in COMPONENT_PREFIXES.items():
        if name.startswith(prefixes):
            return component
    raise ValueError(f"checkpoint tensor {name!r} belongs to no SmolVLA component")


def _remapper(component: str):
    root, prefixes = COMPONENT_PREFIXES[component]

    def remap(name: str) -> str | None:
        if name in SKIPPED or not name.startswith(prefixes):
            return None
        return name[len(root):]

    return remap


def load_smolvla_weights(
    module: nn.Module,
    component: str,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    """Load one component's tensors from the snapshot directory ``path``, strictly:
    every parameter must get one, and
    a tensor stored in fp32 (26 of them: the action, time and state projections
    and the expert's cross K/V) must land in an fp32 parameter, because the
    reference runs them in fp32."""
    root, _ = COMPONENT_PREFIXES[component]
    stacked = COMPONENT_STACKED.get(component, [])
    params = dict(module.named_parameters())
    remap, arrived = _remapper(component), set()

    def tracked(tensors):
        for name, tensor in tensors:
            target = remap(name)
            if target is not None:
                arrived.add(target)
                param = params.get(target)
                if tensor.dtype == torch.float32 and param is not None and param.dtype != torch.float32:
                    raise ValueError(f"{component}: {name!r} is stored fp32 but {target!r} is {param.dtype}")
            yield name, tensor

    loaded = load_weights_into(
        module,
        tracked(iter_safetensors_shards(Path(path), device=device, prefix=root)),
        name_remapper=remap,
        stacked_params=stacked,
    )
    # A fused parameter counts as loaded once *any* of its parts lands (`loader/base.py:143`),
    # so a checkpoint missing one of q/k/v or gate/up would serve those rows at random init.
    missing = sorted(set(params) - loaded) + sorted(
        name.replace(rule.target_suffix, rule.source_suffix)
        for name in params
        for rule in stacked
        if rule.target_suffix in name and name.replace(rule.target_suffix, rule.source_suffix) not in arrived
    )
    if missing:
        raise ValueError(f"{component}: {len(missing)} parameter(s) got no weight, e.g. {missing[:5]}")
    return loaded
