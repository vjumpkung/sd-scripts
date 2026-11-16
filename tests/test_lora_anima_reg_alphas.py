"""Tests for the `network_reg_alphas` network arg of networks/lora_anima.py (tiny fake DiT, CPU only)."""

import pytest
import torch

from networks import lora_anima

DIM = 8
GLOBAL_ALPHA = 4.0

# original names of the Linear layers of the fake DiT (the first block, the second block is analogous)
SELF_ATTN = "blocks.0.self_attn.q_proj"
CROSS_ATTN = "blocks.0.cross_attn.q_proj"
MLP = "blocks.0.mlp.layer1"


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 16)


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(16, 32)


# the class name must be in LoRANetwork.ANIMA_TARGET_REPLACE_MODULE
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.cross_attn = Attention()
        self.mlp = Mlp()


class FakeDiT(torch.nn.Module):
    def __init__(self, num_blocks=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(num_blocks)])


def make_network(**kwargs):
    network = lora_anima.create_network(
        multiplier=1.0,
        network_dim=DIM,
        network_alpha=GLOBAL_ALPHA,
        vae=None,
        text_encoders=[],
        unet=FakeDiT(),
        **kwargs,
    )
    return {lora.original_name: lora for lora in network.unet_loras}


def test_fake_dit_matches_target_replace_modules():
    assert Block.__name__ in lora_anima.LoRANetwork.ANIMA_TARGET_REPLACE_MODULE


def test_reg_alphas_applied_per_regex_and_unmatched_keeps_global_alpha():
    loras = make_network(network_reg_alphas=".*self_attn.*=8,.*cross_attn.*=2")

    assert len(loras) == 6  # 2 blocks x (self_attn, cross_attn, mlp)
    for block in (0, 1):
        self_attn = loras[f"blocks.{block}.self_attn.q_proj"]
        cross_attn = loras[f"blocks.{block}.cross_attn.q_proj"]
        mlp = loras[f"blocks.{block}.mlp.layer1"]

        assert self_attn.alpha.item() == pytest.approx(8.0)
        assert self_attn.lora_dim == DIM
        assert self_attn.scale == pytest.approx(8.0 / DIM)

        assert cross_attn.alpha.item() == pytest.approx(2.0)
        assert cross_attn.lora_dim == DIM
        assert cross_attn.scale == pytest.approx(2.0 / DIM)

        assert mlp.alpha.item() == pytest.approx(GLOBAL_ALPHA)
        assert mlp.lora_dim == DIM
        assert mlp.scale == pytest.approx(GLOBAL_ALPHA / DIM)


def test_reg_alphas_combined_with_reg_dims():
    loras = make_network(
        network_reg_dims=".*self_attn.*=16",
        network_reg_alphas=".*self_attn.*=16,.*mlp.*=2",
    )

    self_attn = loras[SELF_ATTN]
    assert self_attn.lora_dim == 16
    assert self_attn.alpha.item() == pytest.approx(16.0)
    assert self_attn.scale == pytest.approx(1.0)

    # dim falls back to the global dim, alpha comes from reg_alphas
    mlp = loras[MLP]
    assert mlp.lora_dim == DIM
    assert mlp.alpha.item() == pytest.approx(2.0)
    assert mlp.scale == pytest.approx(2.0 / DIM)

    # matched by neither: global dim and global alpha
    cross_attn = loras[CROSS_ATTN]
    assert cross_attn.lora_dim == DIM
    assert cross_attn.alpha.item() == pytest.approx(GLOBAL_ALPHA)
    assert cross_attn.scale == pytest.approx(GLOBAL_ALPHA / DIM)


def test_reg_alphas_alpha_is_independent_of_reg_dims_match():
    # reg_dims changes the rank of self_attn only; reg_alphas matches every module regardless
    loras = make_network(network_reg_dims=".*self_attn.*=16", network_reg_alphas=".*=2")

    assert loras[SELF_ATTN].lora_dim == 16
    assert loras[SELF_ATTN].scale == pytest.approx(2.0 / 16)
    assert loras[CROSS_ATTN].lora_dim == DIM
    assert loras[CROSS_ATTN].scale == pytest.approx(2.0 / DIM)


def test_reg_alphas_first_match_wins():
    loras = make_network(network_reg_alphas=".*self_attn.*=8,.*attn.*=2,.*=1")

    # all three patterns match self_attn; the first one wins
    assert loras[SELF_ATTN].alpha.item() == pytest.approx(8.0)
    # the last two patterns match cross_attn; the earlier one wins
    assert loras[CROSS_ATTN].alpha.item() == pytest.approx(2.0)
    # only the last pattern matches mlp
    assert loras[MLP].alpha.item() == pytest.approx(1.0)


def test_reg_alphas_zero_means_alpha_equals_rank():
    loras = make_network(
        network_reg_dims=".*self_attn.*=16",
        network_reg_alphas=".*self_attn.*=0,.*cross_attn.*=0",
    )

    self_attn = loras[SELF_ATTN]
    assert self_attn.lora_dim == 16
    assert self_attn.alpha.item() == pytest.approx(16.0)
    assert self_attn.scale == pytest.approx(1.0)

    cross_attn = loras[CROSS_ATTN]
    assert cross_attn.lora_dim == DIM
    assert cross_attn.alpha.item() == pytest.approx(float(DIM))
    assert cross_attn.scale == pytest.approx(1.0)

    # unmatched module keeps the global alpha
    assert loras[MLP].scale == pytest.approx(GLOBAL_ALPHA / DIM)


def test_reg_alphas_uses_fullmatch():
    # "self_attn" alone does not fullmatch "blocks.0.self_attn.q_proj"
    loras = make_network(network_reg_alphas="self_attn=8")

    for lora in loras.values():
        assert lora.alpha.item() == pytest.approx(GLOBAL_ALPHA)


def test_without_reg_alphas_all_modules_use_global_alpha():
    loras = make_network()

    assert len(loras) == 6
    for lora in loras.values():
        assert lora.lora_dim == DIM
        assert lora.alpha.item() == pytest.approx(GLOBAL_ALPHA)
        assert lora.scale == pytest.approx(GLOBAL_ALPHA / DIM)


def test_without_reg_alphas_reg_dims_still_uses_global_alpha():
    loras = make_network(network_reg_dims=".*self_attn.*=16")

    assert loras[SELF_ATTN].lora_dim == 16
    assert loras[SELF_ATTN].alpha.item() == pytest.approx(GLOBAL_ALPHA)
    assert loras[SELF_ATTN].scale == pytest.approx(GLOBAL_ALPHA / 16)
