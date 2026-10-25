"""A SmolVLA request's frames, text, state and actions, as LeRobot's deployment prepares them.

Normalization (D03) reads the checkpoint's own processor files: stats are looked
up by feature name, and a feature without them passes through unchanged
(`normalize_processor.py:333`), which is what `smolvla_base` does, its stats
being keyed by pretraining dataset (D08).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

STATE = "observation.state"
ACTION = "action"


class MeanStd(NamedTuple):
    mean: torch.Tensor
    std: torch.Tensor
    eps: float


def prepare_images(frames: Sequence[torch.Tensor], width: int, height: int) -> torch.Tensor:
    """Camera frames (uint8 HxWx3, on the GPU) as SigLIP takes them: ``[n, 3, height, width]`` in [-1, 1].

    As LeRobot's deployment does it (D07): ``/ 255`` on the device, which on CUDA
    is ``u8 * (1/255)`` and not the CPU's division; the frame scaled to fit and
    padded on the left and top with 0 (`vla_utils.py:225-242`); then ``x * 2 - 1``.
    """
    images = []
    for frame in frames:
        image = (frame.float() / 255).permute(2, 0, 1)[None]
        h, w = image.shape[2:]
        if (h, w) != (height, width):
            ratio = max(w / width, h / height)
            rh, rw = int(h / ratio), int(w / ratio)
            image = F.interpolate(image, size=(rh, rw), mode="bilinear", align_corners=False)
            image = F.pad(image, (width - rw, 0, height - rh, 0), value=0)
        images.append(image * 2.0 - 1.0)
    return torch.cat(images)


def tokenize(tokenizer, task: str, max_length: int) -> torch.Tensor:
    """The task's ids. LeRobot ends the task with a newline, truncates at
    ``max_length`` and pads to it; the pads are masked out of attention and
    of the positions, so they are left off here."""
    if not task.endswith("\n"):
        task += "\n"
    return torch.tensor(tokenizer(task, max_length=max_length, truncation=True)["input_ids"])


def load_norm_stats(snapshot: str | Path, processor: str, feature: str) -> MeanStd | None:
    """``feature``'s MEAN_STD stats from ``processor`` (``policy_preprocessor`` or
    ``policy_postprocessor``) in the checkpoint ``snapshot``; None when it has none."""
    snapshot = Path(snapshot)
    steps = json.loads((snapshot / f"{processor}.json").read_text())["steps"]
    step = next(s for s in steps if s["registry_name"] in ("normalizer_processor", "unnormalizer_processor"))
    config = step["config"]
    # LeRobot normalizes an observation only if the step lists it, and the action
    # whenever it has stats (`normalize_processor.py:282-283,302`)
    if feature == STATE and (STATE not in config["features"] or STATE not in (
            config.get("normalize_observation_keys") or (STATE,))):
        return None
    mode = config["norm_map"].get("ACTION" if feature == ACTION else "STATE", "IDENTITY")
    if mode == "IDENTITY":
        return None
    # stats group by the name before the last dot (`normalize_processor.py:230`)
    stats = {k.rsplit(".", 1)[1]: v for k, v in load_file(snapshot / step["state_file"]).items()
             if k.rsplit(".", 1)[0] == feature}
    if not stats:
        return None
    if mode != "MEAN_STD" or not {"mean", "std"} <= stats.keys():
        raise ValueError(f"{feature} is normalized with {mode} from {sorted(stats)}; SmolVLA's port reads mean and std")
    return MeanStd(stats["mean"], stats["std"], config["eps"])


def normalize_state(state: torch.Tensor, stats: MeanStd | None, max_state_dim: int) -> torch.Tensor:
    """``(x - mean) / (std + eps)``, then zero-padded to ``max_state_dim``."""
    if stats is not None:
        state = (state - stats.mean.to(state)) / (stats.std.to(state) + stats.eps)
    return F.pad(state, (0, max_state_dim - state.shape[-1]))


def unnormalize_actions(chunk: torch.Tensor, stats: MeanStd | None, action_dim: int) -> torch.Tensor:
    """The robot's ``action_dim`` columns of a ``max_action_dim``-wide chunk, as ``x * std + mean``."""
    chunk = chunk[..., :action_dim]
    if stats is None:
        return chunk
    return chunk * stats.std.to(chunk) + stats.mean.to(chunk)
