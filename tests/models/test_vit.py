"""Behaviour spec for ``exp.models.vit``.

Translates the approved spec for the 2D axial RoPE Vision Transformer
encoder into executable form. Tests are written against the *spec*, not
any implementation, on real CPU torch tensors with fixed seeds (no
mocking of torch internals, per the project testing strategy).

Two surfaces are pinned:
- ``AxialRoPE`` — the mathematical contract of an orthogonal,
  relative-position rotary embedding (norm preservation, identity at the
  grid origin, translation invariance of rotated dot products) and the
  identity rotation of prepended prefix (CLS) tokens.
- ``VisionTransformer`` — the shape contract over arbitrary leading
  batch dims (now ``n_patches + 1`` output tokens with a leading CLS
  token at index 0), constructor validation, gradient flow into the CLS
  token, eval determinism, and ``torch.compile`` parity (executability
  guarantee).
"""

import pytest
import torch

from exp.models.vit import AxialRoPE, VisionTransformer

# Small ViT config reused across VisionTransformer tests.
# image 8x8, patch 4x4 -> grid 2x2 -> n_patches 4.
# embed_dim 16 / num_heads 2 -> head_dim 8 (8 % 4 == 0, valid).
_IMAGE_SIZE = 8
_PATCH_SIZE = 4
_IN_CHANNELS = 3
_EMBED_DIM = 16
_DEPTH = 2
_NUM_HEADS = 2
_N_PATCHES = 4
# The ViT prepends a single CLS token (index 0), so the output sequence
# length is n_patches + 1.
_N_TOKENS = _N_PATCHES + 1


def _make_vit():
    return VisionTransformer(
        image_size=_IMAGE_SIZE,
        patch_size=_PATCH_SIZE,
        in_channels=_IN_CHANNELS,
        embed_dim=_EMBED_DIM,
        depth=_DEPTH,
        num_heads=_NUM_HEADS,
        dropout=0.0,
        attn_drop=0.0,
    )


class TestAxialRoPEMath:
    def test_preserves_per_token_norm(self):
        # Rotation is orthogonal: each token vector's L2 norm is unchanged.
        torch.manual_seed(0)
        rope = AxialRoPE(head_dim=8, grid_hw=(2, 2))
        q = torch.randn(2, 3, 4, 8)  # (B, heads, n_patches, head_dim)

        out = rope(q)

        torch.testing.assert_close(out.norm(dim=-1), q.norm(dim=-1))

    def test_origin_patch_is_identity(self):
        # Row-major flatten: index 0 is grid (y=0, x=0); zero angle -> no-op.
        torch.manual_seed(0)
        rope = AxialRoPE(head_dim=8, grid_hw=(2, 2))
        q = torch.randn(1, 1, 4, 8)

        out = rope(q)

        torch.testing.assert_close(out[:, :, 0, :], q[:, :, 0, :])

    def test_prefix_token_row_is_identity_rotation(self):
        # With num_prefix_tokens=1 a zero-angle row is prepended for the CLS
        # token, which carries no spatial position and must not be rotated.
        # The sequence is then [prefix, patch_0, ..., patch_{n-1}], so index 0
        # (the prefix row) must pass through unchanged for arbitrary input.
        torch.manual_seed(0)
        rope = AxialRoPE(head_dim=8, grid_hw=(2, 2), num_prefix_tokens=1)
        x = torch.randn(1, 1, 5, 8)  # (B, heads, num_prefix + n_patches, head_dim)

        out = rope(x)

        torch.testing.assert_close(out[:, :, 0, :], x[:, :, 0, :])

    def test_dot_product_invariant_under_joint_translation(self):
        # RoPE relative-position property: <R(p)q, R(p')k> = <q, R(p'-p)k>.
        # It holds for the SAME q,k vectors rotated at different positions, so
        # the inner product depends only on the relative offset p'-p. Placing
        # one shared q0/k0 at every grid cell, two patch pairs separated by the
        # same (dy, dx) must give equal rotated inner products; a different
        # relative offset must give a different one.
        torch.manual_seed(0)
        grid_h, grid_w = 4, 4
        rope = AxialRoPE(head_dim=8, grid_hw=(grid_h, grid_w))

        def idx(y, x):
            return y * grid_w + x  # row-major flatten

        q0 = torch.randn(8)
        k0 = torch.randn(8)
        q = q0.expand(1, 1, grid_h * grid_w, 8).clone()
        k = k0.expand(1, 1, grid_h * grid_w, 8).clone()
        q_rot = rope(q)
        k_rot = rope(k)

        def rotated_dot(p, pprime):
            return (q_rot[0, 0, idx(*p)] * k_rot[0, 0, idx(*pprime)]).sum()

        # Pairs (0,0)->(1,2) and (1,1)->(2,3): both have offset (dy,dx)=(1,1).
        dot_base = rotated_dot((0, 0), (1, 2))
        dot_shift = rotated_dot((1, 1), (2, 3))
        torch.testing.assert_close(dot_base, dot_shift)

        # A different relative offset (0,1) must NOT match offset (1,1).
        dot_other_offset = rotated_dot((0, 0), (0, 1))
        assert not torch.allclose(dot_other_offset, dot_base)

    def test_head_dim_not_multiple_of_4_raises(self):
        # Two axes x even-rotation-pairs require head_dim % 4 == 0; 6 fails.
        with pytest.raises(ValueError, match="4"):
            AxialRoPE(head_dim=6, grid_hw=(2, 2))


class TestVisionTransformerShape:
    def test_no_leading_batch(self):
        # (C, H, W) -> (n_patches + 1, embed_dim): empty leading batch, CLS
        # token prepended.
        model = _make_vit()
        out = model(torch.randn(_IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE))

        assert out.shape == (_N_TOKENS, _EMBED_DIM)

    def test_single_leading_batch(self):
        # (B, C, H, W) -> (B, n_patches + 1, embed_dim).
        model = _make_vit()
        out = model(torch.randn(5, _IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE))

        assert out.shape == (5, _N_TOKENS, _EMBED_DIM)

    def test_two_leading_batch_dims(self):
        # (T, B, C, H, W) -> (T, B, n_patches + 1, embed_dim): flatten/unflatten.
        model = _make_vit()
        out = model(torch.randn(2, 5, _IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE))

        assert out.shape == (2, 5, _N_TOKENS, _EMBED_DIM)

    def test_exposes_n_patches_and_embed_dim(self):
        # Attributes are part of the public contract (used downstream).
        model = _make_vit()

        assert model.n_patches == _N_PATCHES
        assert model.embed_dim == _EMBED_DIM


class TestVisionTransformerValidation:
    def test_head_dim_not_multiple_of_4_raises(self):
        # embed_dim 18 / num_heads 3 -> head_dim 6, not a multiple of 4.
        with pytest.raises(ValueError, match="4"):
            VisionTransformer(
                image_size=8,
                patch_size=4,
                in_channels=3,
                embed_dim=18,
                depth=1,
                num_heads=3,
            )

    def test_image_not_divisible_by_patch_raises(self):
        # image_size 10 is not divisible by patch_size 4 -> no integer grid.
        with pytest.raises(ValueError):
            VisionTransformer(
                image_size=10,
                patch_size=4,
                in_channels=3,
                embed_dim=16,
                depth=1,
                num_heads=2,
            )


class TestVisionTransformerBehaviour:
    def test_backward_populates_some_gradient(self):
        # A scalar loss must flow gradients back to at least one parameter.
        torch.manual_seed(0)
        model = _make_vit()
        x = torch.randn(2, _IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE)

        model(x).sum().backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert any(g.abs().sum() > 0 for g in grads)

    def test_backward_populates_cls_token_gradient(self):
        # The CLS token is the downstream aggregation handle (index 0 of the
        # output). A scalar loss must flow gradient into cls_token, proving the
        # token actually participates in the forward computation rather than
        # being a dangling parameter.
        torch.manual_seed(0)
        model = _make_vit()
        x = torch.randn(2, _IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE)

        model(x).sum().backward()

        assert model.cls_token.grad is not None

    def test_eval_is_deterministic(self):
        # In eval mode dropout is disabled, so repeated forwards must match.
        torch.manual_seed(0)
        model = _make_vit()
        model.eval()
        x = torch.randn(3, _IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE)

        with torch.no_grad():
            first = model(x)
            second = model(x)

        torch.testing.assert_close(first, second)


class TestVisionTransformerCompile:
    # integration-real: exercises the CPU inductor backend. Skips when
    # torch.compile is unavailable/unsupported in the environment, since
    # this only guarantees compile *executability* and eager parity.

    def test_compiled_matches_eager(self):
        torch.manual_seed(0)
        model = _make_vit()
        model.eval()
        x = torch.randn(2, _IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE)

        try:
            compiled = torch.compile(model)
            with torch.no_grad():
                eager_out = model(x)
                compiled_out = compiled(x)
        except Exception as exc:  # compile backend missing/unsupported here
            pytest.skip(f"torch.compile unavailable in this environment: {exc}")

        torch.testing.assert_close(compiled_out, eager_out)
