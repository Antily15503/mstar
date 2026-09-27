"""Numerical-parity tests between mstar's ported GR00T N1.7 DiT and NVIDIA's reference.

``mstar/model/gr00t_n17/components/dit.py`` is a direct port of Isaac-GR00T's
``gr00t/model/modules/dit.py``. The port keeps the reference's class names and
reuses the *same* diffusers primitives (``Attention``, ``FeedForward``,
``Timesteps``, ``TimestepEmbedding``, ``SinusoidalPositionalEmbedding``); what it
drops is plumbing only -- diffusers ``ModelMixin``/``ConfigMixin``/
``register_to_config``, an sm121 SDPA workaround, ``return_all_hidden_states``,
gradient checkpointing and a constructor ``print``. None of that touches
numerics, so the bar here is **bit-exactness** (``torch.equal``), not a
tolerance: both sides run identical kernels over identical fp32 weights. There
is no tolerance to tune, which is the point -- a mismatch of any size is a real
porting bug, never noise.

The reference module is loaded **by file path** with
``importlib.util.spec_from_file_location`` rather than
``import gr00t.model.modules.dit``: that package's ``__init__`` pulls in the
whole Gr00tN1d7Pipeline (transformers, HF hub, flash-attn). ``dit.py`` itself
imports only torch + diffusers, so it loads standalone.

Run with::

    GR00T_REF_DIR=/path/to/Isaac-GR00T \\
        python -m pytest test/modular/test_gr00t_n17_dit_equivalence.py -q

The whole module skips when ``GR00T_REF_DIR`` is unset or does not contain the
reference file. There is deliberately no hardcoded default path.

Coverage:
  * state-dict key/shape sets and parameter count identical to the reference
  * the interleave wiring: even-index blocks cross-attend, odd-index self-attend
    (the single most likely porting bug), for ``num_layers`` even and odd
  * ``TimestepEncoder`` bit-exact, including timestep 0 and the schedule end
  * ``AdaLayerNorm`` bit-exact, plus a guard that the modulation is non-trivial
  * ``BasicTransformerBlock`` bit-exact, self-attention case
  * ``BasicTransformerBlock`` bit-exact, cross-attention case with a non-trivial
    ``encoder_attention_mask`` (masking conventions are the second most likely bug)
  * ``DiT`` end-to-end bit-exact, batch=2, with and without interleaving
  * ``DiT`` bit-exact with a single context token and with non-contiguous inputs
  * ``AlternateVLDiT`` end-to-end bit-exact with a mixed ``image_mask`` and a
    padded ``backbone_attention_mask``, for ``attend_text_every_n_blocks``
    in {1, 2, 3}
  * ``SelfAttentionTransformer`` (the VL bridge) bit-exact
  * behavioural, no reference: image-masked context tokens affect the output;
    tokens excluded by ``backbone_attention_mask`` are bit-irrelevant; both the
    text and the image branch of the alternation are genuinely exercised
  * output shape ``(B, T, output_dim)``, documenting ``inner_dim != output_dim``
  * output dtype/device follow the input; repeated calls are bit-identical; one
    batch row cannot move another
  * bf16 forward produces no NaN/Inf (serving dtype is bfloat16) -- smoke only
  * ``AlternateVLDiT`` rejects a missing ``image_mask`` /
    ``backbone_attention_mask`` / ``interleave_self_attention=False`` cleanly
  * a divergence the port introduces on purpose: the reference ``DiT.forward``
    silently drops ``encoder_attention_mask``; ours honours it
"""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytest.importorskip("diffusers")

sys.path.insert(0, ".")

REF_ENV_VAR = "GR00T_REF_DIR"
_REF_ROOT = os.environ.get(REF_ENV_VAR)
_REF_DIT = Path(_REF_ROOT, "gr00t", "model", "modules", "dit.py") if _REF_ROOT else None

pytestmark = pytest.mark.skipif(
    _REF_DIT is None or not _REF_DIT.is_file(),
    reason=(
        f"set {REF_ENV_VAR} to an Isaac-GR00T checkout "
        f"(needs <{REF_ENV_VAR}>/gr00t/model/modules/dit.py) to run GR00T DiT parity tests"
    ),
)

# Tiny CPU config. inner_dim = num_attention_heads * attention_head_dim = 2 * 16
# = 32, deliberately different from output_dim (24) so a test that confused the
# two would fail -- the real checkpoint has the same asymmetry (1536 vs 1024).
# num_layers is 4, not 2, because with attend_text_every_n_blocks=2 the cross
# blocks are the even indices and idx % 4 == 0 selects the TEXT branch while
# idx % 4 == 2 selects the IMAGE branch; 4 layers is the smallest stack that
# contains both. cross_attention_dim (48) differs from inner_dim so the
# self-vs-cross wiring is visible in to_k.in_features. dropout=0.0 and
# final_dropout=False remove the only stochastic ops, and every model is also
# put in .eval() -- belt and braces, since dropout at p=0.0 is an identity but
# .eval() is what the serving path actually relies on.
TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "output_dim": 24,
    "num_layers": 4,
    "dropout": 0.0,
    "attention_bias": True,
    "activation_fn": "gelu-approximate",
    "upcast_attention": False,
    "norm_type": "ada_norm",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-5,
    "max_num_positional_embeddings": 512,
    "final_dropout": False,
    "positional_embeddings": "sinusoidal",
    "interleave_self_attention": True,
    "cross_attention_dim": 48,
}
INNER_DIM = TINY["num_attention_heads"] * TINY["attention_head_dim"]
OUTPUT_DIM = TINY["output_dim"]
CROSS_DIM = TINY["cross_attention_dim"]
SEED = 1234


# ----------------------------------------------------------------------
# Module loading
# ----------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def ref_module() -> ModuleType:
    """Load the reference ``dit.py`` by path, bypassing the gr00t package __init__."""
    spec = importlib.util.spec_from_file_location("gr00t_ref_dit", str(_REF_DIT))
    module = importlib.util.module_from_spec(spec)
    # Registered so dataclass/pickle machinery inside torch can resolve the module.
    sys.modules["gr00t_ref_dit"] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def our_module() -> ModuleType:
    from mstar.model.gr00t_n17.components import dit

    return dit


def build_pair(class_name: str, **kwargs):
    """Build (reference, ours) of ``class_name``, weight-shared and in eval mode.

    Both are seeded identically before construction, then the reference's
    ``state_dict`` is loaded into ours with ``strict=True``: the port keeps every
    parameter name, so no remapping helper is needed and ``strict=True`` is the
    assertion that this is still true.
    """
    torch.manual_seed(SEED)
    ref = getattr(ref_module(), class_name)(**kwargs)
    torch.manual_seed(SEED)
    ours = getattr(our_module(), class_name)(**kwargs)
    ours.load_state_dict(ref.state_dict(), strict=True)
    return ref.eval(), ours.eval()


def shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in module.state_dict().items()}


def assert_bit_exact(ours: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    assert ours.shape == ref.shape, f"{what}: shape {tuple(ours.shape)} != reference {tuple(ref.shape)}"
    if torch.equal(ours, ref):
        return
    delta = (ours.double() - ref.double()).abs()
    raise AssertionError(
        f"{what}: not bit-exact against the reference. "
        f"max abs delta {delta.max().item():.3e}, mean abs delta {delta.mean().item():.3e}, "
        f"{int((ours != ref).sum())}/{ours.numel()} elements differ"
    )


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------


def dit_inputs(batch: int = 2, action_len: int = 5, ctx_len: int = 7, seed: int = SEED):
    torch.manual_seed(seed)
    hidden = torch.randn(batch, action_len, INNER_DIM)
    context = torch.randn(batch, ctx_len, CROSS_DIM)
    timestep = torch.randint(0, 1000, (batch,))
    return hidden, context, timestep


def vl_masks(batch: int = 2, ctx_len: int = 7):
    """Mixed image/text mask plus right-padding, so both derived masks are non-trivial.

    Row 0: tokens 0-2 text, 3-5 image, 6 padding.
    Row 1: token 0 text, 1-4 image, 5-6 padding.
    Every row keeps at least one live text token and one live image token, so
    neither derived mask has an all-masked attention row (which would make
    softmax produce NaN in *both* implementations and prove nothing).
    """
    assert ctx_len == 7 and batch == 2, "vl_masks hardcodes the layout it documents"
    image_mask = torch.tensor(
        [
            [False, False, False, True, True, True, False],
            [False, True, True, True, True, False, False],
        ]
    )
    backbone_mask = torch.tensor(
        [
            [True, True, True, True, True, True, False],
            [True, True, True, True, True, False, False],
        ]
    )
    return image_mask, backbone_mask


# ----------------------------------------------------------------------
# 1. Structure
# ----------------------------------------------------------------------


def test_dit_state_dict_keys_shapes_and_parameter_count_match_the_reference():
    torch.manual_seed(SEED)
    ref = ref_module().DiT(**TINY)
    torch.manual_seed(SEED)
    ours = our_module().DiT(**TINY)

    ref_shapes, our_shapes = shapes(ref), shapes(ours)
    assert set(our_shapes) == set(ref_shapes), (
        f"state-dict keys diverge. only in ours: {sorted(set(our_shapes) - set(ref_shapes))}; "
        f"only in reference: {sorted(set(ref_shapes) - set(our_shapes))}"
    )
    mismatched = {k: (our_shapes[k], ref_shapes[k]) for k in ref_shapes if our_shapes[k] != ref_shapes[k]}
    assert not mismatched, f"state-dict shapes diverge (ours, reference): {mismatched}"

    ours_n = sum(p.numel() for p in ours.parameters())
    ref_n = sum(p.numel() for p in ref.parameters())
    assert ours_n == ref_n, f"parameter count {ours_n} != reference {ref_n} (delta {ours_n - ref_n})"


def test_alternate_vl_dit_state_dict_matches_the_reference():
    torch.manual_seed(SEED)
    ref = ref_module().AlternateVLDiT(**TINY, attend_text_every_n_blocks=2)
    torch.manual_seed(SEED)
    ours = our_module().AlternateVLDiT(**TINY, attend_text_every_n_blocks=2)
    assert shapes(ours) == shapes(ref), "AlternateVLDiT state-dict diverges from the reference"


@pytest.mark.parametrize("num_layers", [3, 4, 5])
def test_even_index_blocks_cross_attend_and_odd_index_blocks_self_attend(num_layers):
    """The interleave wiring, read off the projection widths rather than the config.

    ``attn1.to_k.in_features`` is ``cross_attention_dim`` for a cross-attention
    block and ``inner_dim`` for a self-attention one, so this catches an
    off-by-one in the ``idx % 2`` predicate even when shapes still load.
    """
    cfg = {**TINY, "num_layers": num_layers}
    ref, ours = build_pair("DiT", **cfg)
    for idx in range(num_layers):
        expected = INNER_DIM if idx % 2 == 1 else CROSS_DIM
        assert ours.transformer_blocks[idx].attn1.to_k.in_features == expected, (
            f"block {idx}: to_k.in_features == {ours.transformer_blocks[idx].attn1.to_k.in_features}, "
            f"expected {expected} ({'self' if idx % 2 else 'cross'}-attention)"
        )
        assert (
            ours.transformer_blocks[idx].attn1.to_k.in_features
            == ref.transformer_blocks[idx].attn1.to_k.in_features
        ), f"block {idx} attention kind disagrees with the reference"


def test_interleave_disabled_makes_every_block_cross_attend_in_both_implementations():
    cfg = {**TINY, "interleave_self_attention": False}
    ref, ours = build_pair("DiT", **cfg)
    for idx in range(cfg["num_layers"]):
        assert ours.transformer_blocks[idx].attn1.to_k.in_features == CROSS_DIM, f"block {idx} is not cross-attention"
        assert (
            ours.transformer_blocks[idx].attn1.to_k.in_features
            == ref.transformer_blocks[idx].attn1.to_k.in_features
        )


# ----------------------------------------------------------------------
# 2. Leaf modules
# ----------------------------------------------------------------------


def test_timestep_encoder_is_bit_exact_across_the_schedule():
    ref, ours = build_pair("TimestepEncoder", embedding_dim=INNER_DIM)
    # 0 and 999 are the endpoints of the released 1000-step schedule; 1 and 500
    # sit next to the low end and in the middle.
    timesteps = torch.tensor([0, 1, 500, 999])
    with torch.no_grad():
        assert_bit_exact(ours(timesteps), ref(timesteps), "TimestepEncoder(t=[0,1,500,999])")


def test_ada_layer_norm_is_bit_exact_and_its_modulation_is_non_trivial():
    ref, ours = build_pair("AdaLayerNorm", embedding_dim=INNER_DIM)
    torch.manual_seed(SEED + 1)
    x = torch.randn(2, 5, INNER_DIM)
    temb = torch.randn(2, INNER_DIM)
    with torch.no_grad():
        out = ours(x, temb)
        assert_bit_exact(out, ref(x, temb), "AdaLayerNorm")
        # A port that normalised x and ignored temb would still be bit-exact
        # against nothing, so pin the conditioning down: a different temb must
        # move the output.
        other = ours(x, torch.randn(2, INNER_DIM))
    delta = (out - other).abs().max().item()
    assert delta > 1e-4, f"AdaLayerNorm output is insensitive to temb (max abs delta {delta:.3e}); modulation is dead"


def test_basic_transformer_block_is_bit_exact_in_the_self_attention_case():
    kwargs = {
        "dim": INNER_DIM,
        "num_attention_heads": TINY["num_attention_heads"],
        "attention_head_dim": TINY["attention_head_dim"],
        "dropout": 0.0,
        "cross_attention_dim": None,
        "activation_fn": TINY["activation_fn"],
        "attention_bias": True,
        "norm_type": "ada_norm",
        "norm_elementwise_affine": False,
        "final_dropout": False,
        "positional_embeddings": "sinusoidal",
        "num_positional_embeddings": 512,
    }
    ref, ours = build_pair("BasicTransformerBlock", **kwargs)
    torch.manual_seed(SEED + 2)
    hidden = torch.randn(2, 5, INNER_DIM)
    temb = torch.randn(2, INNER_DIM)
    with torch.no_grad():
        assert_bit_exact(
            ours(hidden, temb=temb),
            ref(hidden, temb=temb),
            "BasicTransformerBlock (self-attention)",
        )


def test_basic_transformer_block_is_bit_exact_when_cross_attending_under_a_mask():
    """Masking conventions are the likeliest silent divergence after the wiring.

    The mask here is non-trivial and ragged (different live lengths per row), so
    a flipped polarity or a mask applied to the wrong axis changes the numbers.
    """
    kwargs = {
        "dim": INNER_DIM,
        "num_attention_heads": TINY["num_attention_heads"],
        "attention_head_dim": TINY["attention_head_dim"],
        "dropout": 0.0,
        "cross_attention_dim": CROSS_DIM,
        "activation_fn": TINY["activation_fn"],
        "attention_bias": True,
        "norm_type": "ada_norm",
        "norm_elementwise_affine": False,
        "final_dropout": False,
        "positional_embeddings": "sinusoidal",
        "num_positional_embeddings": 512,
    }
    ref, ours = build_pair("BasicTransformerBlock", **kwargs)
    torch.manual_seed(SEED + 3)
    hidden = torch.randn(2, 5, INNER_DIM)
    context = torch.randn(2, 7, CROSS_DIM)
    temb = torch.randn(2, INNER_DIM)
    mask = torch.tensor(
        [
            [True, True, True, True, False, False, False],
            [True, False, True, True, True, True, False],
        ]
    )
    with torch.no_grad():
        out = ours(hidden, encoder_hidden_states=context, encoder_attention_mask=mask, temb=temb)
        assert_bit_exact(
            out,
            ref(hidden, encoder_hidden_states=context, encoder_attention_mask=mask, temb=temb),
            "BasicTransformerBlock (cross-attention, masked)",
        )
        unmasked = ours(hidden, encoder_hidden_states=context, encoder_attention_mask=None, temb=temb)
    delta = (out - unmasked).abs().max().item()
    assert delta > 1e-5, f"encoder_attention_mask had no effect (max abs delta {delta:.3e}); the mask is being ignored"


def test_self_attention_transformer_bridge_is_bit_exact():
    kwargs = {
        "num_attention_heads": TINY["num_attention_heads"],
        "attention_head_dim": TINY["attention_head_dim"],
        "num_layers": TINY["num_layers"],
        "dropout": 0.0,
        "attention_bias": True,
        "activation_fn": TINY["activation_fn"],
        "upcast_attention": False,
        "max_num_positional_embeddings": 512,
        "final_dropout": False,
        "positional_embeddings": "sinusoidal",
    }
    ref, ours = build_pair("SelfAttentionTransformer", **kwargs)
    torch.manual_seed(SEED + 4)
    hidden = torch.randn(2, 9, INNER_DIM)
    with torch.no_grad():
        assert_bit_exact(ours(hidden), ref(hidden), "SelfAttentionTransformer")


# ----------------------------------------------------------------------
# 3. End-to-end parity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("interleave", [True, False])
def test_dit_forward_is_bit_exact_end_to_end(interleave):
    cfg = {**TINY, "interleave_self_attention": interleave}
    ref, ours = build_pair("DiT", **cfg)
    hidden, context, timestep = dit_inputs(batch=2)
    with torch.no_grad():
        assert_bit_exact(
            ours(hidden, context, timestep),
            ref(hidden, context, timestep),
            f"DiT end-to-end (interleave_self_attention={interleave})",
        )


@pytest.mark.parametrize("timestep_value", [0, 999])
def test_dit_forward_is_bit_exact_at_both_ends_of_the_schedule(timestep_value):
    """t=0 exercises the sin/cos embedding at the origin, t=999 at the schedule end."""
    ref, ours = build_pair("DiT", **TINY)
    hidden, context, _ = dit_inputs(batch=2)
    timestep = torch.full((2,), timestep_value, dtype=torch.long)
    with torch.no_grad():
        assert_bit_exact(
            ours(hidden, context, timestep),
            ref(hidden, context, timestep),
            f"DiT end-to-end (t={timestep_value})",
        )


def test_dit_forward_is_bit_exact_with_a_single_context_token():
    """S=1 is the degenerate cross-attention case: softmax over one key."""
    ref, ours = build_pair("DiT", **TINY)
    hidden, _, timestep = dit_inputs(batch=2)
    torch.manual_seed(SEED + 5)
    context = torch.randn(2, 1, CROSS_DIM)
    with torch.no_grad():
        assert_bit_exact(
            ours(hidden, context, timestep),
            ref(hidden, context, timestep),
            "DiT end-to-end (single context token)",
        )


def test_dit_forward_is_bit_exact_on_non_contiguous_inputs():
    """Both sides call .contiguous() on entry; check that is actually reached.

    A transposed view has the right shape and the wrong strides. If either side
    forgot the ``.contiguous()`` the kernels would still run, so this is about
    the two agreeing on layout handling, which a fused attention path can
    otherwise make visible.
    """
    ref, ours = build_pair("DiT", **TINY)
    torch.manual_seed(SEED + 6)
    hidden = torch.randn(2, INNER_DIM, 5).transpose(1, 2)
    context = torch.randn(2, CROSS_DIM, 7).transpose(1, 2)
    timestep = torch.tensor([3, 700])
    assert not hidden.is_contiguous() and not context.is_contiguous()
    with torch.no_grad():
        assert_bit_exact(
            ours(hidden, context, timestep),
            ref(hidden, context, timestep),
            "DiT end-to-end (non-contiguous inputs)",
        )


@pytest.mark.parametrize("attend_text_every_n_blocks", [1, 2, 3])
def test_alternate_vl_dit_forward_is_bit_exact_under_mixed_and_padded_masks(attend_text_every_n_blocks):
    ref, ours = build_pair("AlternateVLDiT", **TINY, attend_text_every_n_blocks=attend_text_every_n_blocks)
    hidden, context, timestep = dit_inputs(batch=2, ctx_len=7)
    image_mask, backbone_mask = vl_masks()
    with torch.no_grad():
        got = ours(
            hidden,
            context,
            timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_mask,
        )
        want = ref(
            hidden,
            context,
            timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_mask,
        )
    assert_bit_exact(got, want, f"AlternateVLDiT end-to-end (attend_text_every_n_blocks={attend_text_every_n_blocks})")


def test_the_port_honours_encoder_attention_mask_on_the_plain_dit_where_the_reference_drops_it():
    """A deliberate, documented divergence -- not a numerical-parity claim.

    The reference ``DiT.forward`` accepts ``encoder_attention_mask`` and then
    hardcodes ``encoder_attention_mask=None`` on every block call, so the
    argument is dead. The port forwards it. This test pins that difference down
    so it cannot drift unnoticed: if a future change makes the port ignore the
    mask too, or makes the reference honour it, this test fails and someone has
    to look. Nothing in the GR00T N1.7 serving path passes this argument to the
    plain ``DiT`` (``AlternateVLDiT`` builds its own masks), so the divergence is
    currently unreachable in production.
    """
    ref, ours = build_pair("DiT", **TINY)
    hidden, context, timestep = dit_inputs(batch=2)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False, False],
            [True, True, False, False, False, False, False],
        ]
    )
    with torch.no_grad():
        ref_masked = ref(hidden, context, timestep, encoder_attention_mask=mask)
        ref_unmasked = ref(hidden, context, timestep, encoder_attention_mask=None)
        our_masked = ours(hidden, context, timestep, encoder_attention_mask=mask)
        our_unmasked = ours(hidden, context, timestep, encoder_attention_mask=None)

    assert torch.equal(ref_masked, ref_unmasked), (
        "the reference DiT now reacts to encoder_attention_mask; the port's behaviour is no longer a "
        "superset and this divergence needs re-reviewing"
    )
    assert_bit_exact(our_unmasked, ref_unmasked, "DiT end-to-end (encoder_attention_mask=None)")
    delta = (our_masked - our_unmasked).abs().max().item()
    assert delta > 1e-5, (
        f"the port no longer honours encoder_attention_mask on the plain DiT "
        f"(max abs delta {delta:.3e}); it has regressed to the reference's dead argument"
    )


# ----------------------------------------------------------------------
# 4. Behavioural properties (no reference involved)
# ----------------------------------------------------------------------


def test_alternate_vl_dit_attends_image_tokens_and_ignores_tokens_outside_the_backbone_mask():
    """The masks must do two opposite things, and both are checked.

    Live image tokens must influence the output (otherwise the image branch is
    dead). Tokens excluded by ``backbone_attention_mask`` must be *bit*-
    irrelevant: a boolean SDPA mask drives their softmax weight to exactly zero,
    so any drift at all means padding is leaking into the actions.
    """
    _, ours = build_pair("AlternateVLDiT", **TINY, attend_text_every_n_blocks=2)
    hidden, context, timestep = dit_inputs(batch=2, ctx_len=7)
    image_mask, backbone_mask = vl_masks()
    kwargs = {"image_mask": image_mask, "backbone_attention_mask": backbone_mask}

    torch.manual_seed(SEED + 7)
    noise = torch.randn_like(context) * 5.0
    live_image = (image_mask & backbone_mask).unsqueeze(-1)
    live_text = ((~image_mask) & backbone_mask).unsqueeze(-1)
    dead = (~backbone_mask).unsqueeze(-1)
    assert live_image.any() and live_text.any() and dead.any(), "vl_masks no longer exercises all three regions"

    with torch.no_grad():
        base = ours(hidden, context, timestep, **kwargs)
        image_perturbed = ours(hidden, context + noise * live_image, timestep, **kwargs)
        text_perturbed = ours(hidden, context + noise * live_text, timestep, **kwargs)
        pad_perturbed = ours(hidden, context + noise * dead, timestep, **kwargs)

    image_delta = (image_perturbed - base).abs().max().item()
    text_delta = (text_perturbed - base).abs().max().item()
    assert image_delta > 1e-4, f"perturbing live image tokens changed nothing (max abs delta {image_delta:.3e})"
    assert text_delta > 1e-4, f"perturbing live text tokens changed nothing (max abs delta {text_delta:.3e})"

    if not torch.equal(pad_perturbed, base):
        pad_delta = (pad_perturbed - base).abs().max().item()
        raise AssertionError(
            f"tokens masked out by backbone_attention_mask leaked into the output "
            f"(max abs delta {pad_delta:.3e}); they must be bit-irrelevant"
        )


def test_rows_of_a_batch_do_not_leak_into_each_other():
    """Nothing in the DiT mixes across the batch axis, so one row's inputs must
    not move another row's output by a single bit.

    This is the batch-independence property that matters and it can be stated
    bit-exactly, unlike a B=1-vs-B=3 comparison: changing the batch *size*
    changes the CPU kernels' reduction order (observed ~4e-7 here), which is a
    torch artifact, not a claim about this port. Perturbing one row in place
    keeps the shapes and therefore the kernels identical, so any drift in the
    other rows is a real bug (a mask broadcast over the wrong axis, or state
    leaking between rows).
    """
    _, ours = build_pair("AlternateVLDiT", **TINY, attend_text_every_n_blocks=2)
    hidden, context, timestep = dit_inputs(batch=2, ctx_len=7)
    image_mask, backbone_mask = vl_masks()
    kwargs = {"image_mask": image_mask, "backbone_attention_mask": backbone_mask}

    torch.manual_seed(SEED + 8)
    perturbed_hidden = hidden.clone()
    perturbed_hidden[0] += torch.randn_like(hidden[0]) * 3.0
    perturbed_context = context.clone()
    perturbed_context[0] += torch.randn_like(context[0]) * 3.0

    with torch.no_grad():
        base = ours(hidden, context, timestep, **kwargs)
        moved = ours(perturbed_hidden, perturbed_context, timestep, **kwargs)

    assert_bit_exact(moved[1:], base[1:], "batch row 1 after perturbing only row 0")
    row0_delta = (moved[0] - base[0]).abs().max().item()
    assert row0_delta > 1e-4, f"perturbing row 0 did not move row 0 (max abs delta {row0_delta:.3e})"


def test_alternate_vl_dit_output_shape_is_batch_by_action_len_by_output_dim():
    """``inner_dim`` (32 here, 1536 in the checkpoint) is not ``output_dim``
    (24 here, 1024 in the checkpoint) -- ``proj_out_2`` changes width on the way
    out, and the DiT's contract is the *output_dim* one."""
    _, ours = build_pair("AlternateVLDiT", **TINY, attend_text_every_n_blocks=2)
    hidden, context, timestep = dit_inputs(batch=2, action_len=5, ctx_len=7)
    image_mask, backbone_mask = vl_masks()
    with torch.no_grad():
        out = ours(hidden, context, timestep, image_mask=image_mask, backbone_attention_mask=backbone_mask)
    assert out.shape == (2, 5, OUTPUT_DIM), f"expected (2, 5, {OUTPUT_DIM}), got {tuple(out.shape)}"
    assert INNER_DIM != OUTPUT_DIM, "the test config no longer distinguishes inner_dim from output_dim"
    assert out.dtype == hidden.dtype and out.device == hidden.device


def test_repeated_forwards_are_bit_identical():
    """No leftover dropout or other nondeterminism in eval mode."""
    _, ours = build_pair("AlternateVLDiT", **TINY, attend_text_every_n_blocks=2)
    hidden, context, timestep = dit_inputs(batch=2, ctx_len=7)
    image_mask, backbone_mask = vl_masks()
    kwargs = {"image_mask": image_mask, "backbone_attention_mask": backbone_mask}
    with torch.no_grad():
        first = ours(hidden, context, timestep, **kwargs)
        second = ours(hidden, context, timestep, **kwargs)
    assert_bit_exact(second, first, "AlternateVLDiT called twice")


def test_alternate_vl_dit_rejects_missing_masks_and_disabled_interleaving():
    _, ours = build_pair("AlternateVLDiT", **TINY, attend_text_every_n_blocks=2)
    hidden, context, timestep = dit_inputs(batch=2, ctx_len=7)
    image_mask, backbone_mask = vl_masks()
    with torch.no_grad():
        with pytest.raises(ValueError, match="image_mask"):
            ours(hidden, context, timestep, backbone_attention_mask=backbone_mask)
        with pytest.raises(ValueError, match="backbone_attention_mask"):
            ours(hidden, context, timestep, image_mask=image_mask)

    _, no_interleave = build_pair(
        "AlternateVLDiT",
        **{**TINY, "interleave_self_attention": False},
        attend_text_every_n_blocks=2,
    )
    with torch.no_grad(), pytest.raises(ValueError, match="interleave_self_attention"):
        no_interleave(hidden, context, timestep, image_mask=image_mask, backbone_attention_mask=backbone_mask)


def test_bfloat16_forward_produces_no_nan_or_inf():
    """Serving runs the action head in bfloat16.

    Smoke only: bf16 on CPU has its own reduction order, so no parity tolerance
    is asserted here -- only that the graph survives the dtype and does not
    produce NaN/Inf (e.g. from a masked softmax row or an overflowing SiLU).
    """
    _, ours = build_pair("AlternateVLDiT", **TINY, attend_text_every_n_blocks=2)
    ours = ours.to(torch.bfloat16)
    hidden, context, timestep = dit_inputs(batch=2, ctx_len=7)
    image_mask, backbone_mask = vl_masks()
    with torch.no_grad():
        out = ours(
            hidden.to(torch.bfloat16),
            context.to(torch.bfloat16),
            timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_mask,
        )
    assert out.dtype == torch.bfloat16, f"output dtype {out.dtype}, expected bfloat16"
    assert torch.isfinite(out).all(), f"bfloat16 forward produced {int((~torch.isfinite(out)).sum())} non-finite values"
