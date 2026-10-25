"""SmolVLA's configs, read off two ``config.json`` files: the policy's own
(``lerobot/smolvla_base``: the expert, the action chunk, preprocessing) and its
backbone's (``HuggingFaceTB/SmolVLM2-500M-Video-Instruct``: text and vision dims).

Fields the vision json leaves out keep transformers' ``SmolVLMVisionConfig``
defaults (12 layers, MLP 3072, ``gelu_pytorch_tanh``, eps 1e-6), which is what
the reference builds from the same json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


def _read(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    return json.loads(path.read_text())


def _known(cls, raw: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def expert_intermediate_size(hidden_size: int, ffn_dim_multiplier: int = 4, multiple_of: int = 256) -> int:
    """The expert's MLP width, as LeRobot derives it (``get_intermediate_size``): 720 -> 2048."""
    size = ffn_dim_multiplier * int(2 * hidden_size / 3)
    return multiple_of * ((size + multiple_of - 1) // multiple_of)


@dataclass
class SmolVLAVisionConfig:
    hidden_size: int
    image_size: int
    patch_size: int
    num_attention_heads: int
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass
class SmolVLATextConfig:
    """The SmolLM2 text model, cut to its first ``num_hidden_layers`` (= ``num_vlm_layers``)."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int


@dataclass
class SmolVLAExpertConfig:
    """The action expert: a narrower copy of the text layers, one per VLM layer.

    Every ``self_attn_every_n_layers``-th layer (from 0) self-attends the prefix
    plus the actions; the rest cross-attend the prefix only. Head counts are the
    text model's, so its queries are 15 x 64 = 960 wide on a 720-wide stream.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    self_attn_every_n_layers: int

    @property
    def cross_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if i % self.self_attn_every_n_layers)


@dataclass
class SmolVLAConfig:
    text: SmolVLATextConfig
    vision: SmolVLAVisionConfig
    expert: SmolVLAExpertConfig

    scale_factor: int  # the connector's pixel shuffle
    camera_names: tuple[str, ...]
    state_dim: int
    action_dim: int
    chunk_size: int = 50
    n_action_steps: int = 50
    num_steps: int = 10
    max_state_dim: int = 32
    max_action_dim: int = 32
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    tokenizer_max_length: int = 48
    empty_cameras: int = 0
    min_period: float = 4e-3
    max_period: float = 4.0
    # Not the backbone json's 100000: LeRobot's `apply_rope` defaults to 10000
    # and no call site passes another (`smolvlm_with_expert.py:41`), for the VLM
    # and the expert alike, so that is the base SmolVLA was trained and runs with.
    rope_theta: float = 10_000.0

    @property
    def tokens_per_image(self) -> int:
        return self.vision.num_patches // self.scale_factor**2

    @property
    def connector_in_features(self) -> int:
        return self.vision.hidden_size * self.scale_factor**2

    @classmethod
    def from_hf(cls, path: str | Path, vlm_path: str | Path) -> "SmolVLAConfig":
        raw = _read(path)
        vlm = _read(vlm_path)
        tc = vlm["text_config"]
        _check_supported(raw, tc, vlm["vision_config"])
        text = SmolVLATextConfig(**_known(SmolVLATextConfig, {**tc, "num_hidden_layers": raw["num_vlm_layers"]}))
        width = int(text.hidden_size * raw["expert_width_multiplier"])
        expert = SmolVLAExpertConfig(
            hidden_size=width,
            intermediate_size=expert_intermediate_size(width),
            num_hidden_layers=text.num_hidden_layers,
            self_attn_every_n_layers=raw["self_attn_every_n_layers"],
        )
        features = raw["input_features"]
        return cls(
            text=text,
            vision=SmolVLAVisionConfig(**_known(SmolVLAVisionConfig, vlm["vision_config"])),
            expert=expert,
            scale_factor=vlm["scale_factor"],
            camera_names=tuple(k for k, v in features.items() if v["type"] == "VISUAL"),
            state_dim=features["observation.state"]["shape"][0],
            action_dim=raw["output_features"]["action"]["shape"][0],
            **_known(cls, {**raw, "resize_imgs_with_padding": tuple(raw["resize_imgs_with_padding"])}),
        )


# The one reference variant this port implements; any other value is a
# different forward, so it fails here rather than running the wrong one.
_REQUIRED = {
    "type": "smolvla",
    "attention_mode": "cross_attn",
    "self_attn_every_n_layers": 2,
    "add_image_special_tokens": False,
    "use_cache": True,
    "adapt_to_pi_aloha": False,
    "use_delta_joint_actions_aloha": False,
}


def _check_supported(raw: dict, text_config: dict, vision_config: dict) -> None:
    bad = {k: raw.get(k) for k, v in _REQUIRED.items() if raw.get(k, v) != v}
    if raw.get("num_expert_layers", 0) > 0 and raw["num_expert_layers"] != raw["num_vlm_layers"]:
        # LeRobot then shares each expert layer across several VLM layers
        bad["num_expert_layers"] = raw["num_expert_layers"]
    if not 0 < raw["num_vlm_layers"] <= text_config["num_hidden_layers"]:
        bad["num_vlm_layers"] = raw["num_vlm_layers"]
    if raw.get("n_action_steps", 0) > raw.get("chunk_size", 50):
        bad["n_action_steps"] = raw["n_action_steps"]
    if raw.get("resize_imgs_with_padding") is None:
        # LeRobot then skips the resize and feeds images at their own size
        bad["resize_imgs_with_padding"] = None
    elif tuple(raw["resize_imgs_with_padding"]) != (vision_config["image_size"],) * 2:
        # SmolVLM buckets other sizes' positions; SigLIP's tower (D09) only matches at its own size
        bad["resize_imgs_with_padding"] = raw["resize_imgs_with_padding"]
    if not any(v["type"] == "VISUAL" for v in raw["input_features"].values()):
        bad["input_features"] = "no camera"
    if (raw.get("rtc_config") or {}).get("enabled"):
        # real-time chunking guides each Euler step with the previous chunk (`modeling_smolvla.py:818`)
        bad["rtc_config"] = raw["rtc_config"]
    if bad:
        raise ValueError(f"unsupported SmolVLA variant: {bad}")
