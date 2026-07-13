import argparse
import re

import pytest

from library.anima_lora_args import (
    ANIMA_LORA_COMPONENT_PATTERNS,
    add_anima_lora_component_arguments,
    apply_anima_lora_component_args,
)


COMPONENT_NAMES = [name for name, _ in ANIMA_LORA_COMPONENT_PATTERNS]


def make_component_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_args", nargs="*")
    add_anima_lora_component_arguments(parser)
    return parser


def test_component_parser_exposes_all_twelve_options():
    parser = make_component_parser()
    command = []
    for component in COMPONENT_NAMES:
        command.extend((f"--{component}_dim", "8", f"--{component}_lr", "1e-4"))

    args = parser.parse_args(command)

    for component in COMPONENT_NAMES:
        assert getattr(args, f"{component}_dim") == 8
        assert getattr(args, f"{component}_lr") == pytest.approx(1e-4)


@pytest.mark.parametrize("option,value", [("--unet_mlp_dim", "-1"), ("--te_mlp_lr", "-1e-4")])
def test_component_parser_rejects_negative_values(option, value):
    parser = make_component_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([option, value])


def test_component_options_are_scoped_and_translated_to_network_args():
    parser = make_component_parser()
    args = parser.parse_args(
        [
            "--unet_self_attn_dim",
            "16",
            "--unet_self_attn_lr",
            "1e-4",
            "--te_self_attn_dim",
            "8",
            "--te_self_attn_lr",
            "1e-5",
        ]
    )

    apply_anima_lora_component_args(args)

    assert args.network_args == [
        r"network_reg_dims=blocks\.\d+\.self_attn\..*=16,layers\.\d+\.self_attn\..*=8",
        r"network_reg_lrs=blocks\.\d+\.self_attn\..*=0.0001,layers\.\d+\.self_attn\..*=1e-05",
    ]

    patterns = dict(ANIMA_LORA_COMPONENT_PATTERNS)
    assert re.fullmatch(patterns["unet_self_attn"], "blocks.0.self_attn.q_proj")
    assert not re.fullmatch(patterns["unet_self_attn"], "layers.0.self_attn.q_proj")
    assert re.fullmatch(patterns["te_self_attn"], "layers.0.self_attn.q_proj")
    assert not re.fullmatch(patterns["te_self_attn"], "blocks.0.self_attn.q_proj")


def test_explicit_regexes_keep_priority_over_component_options():
    parser = make_component_parser()
    args = parser.parse_args(
        [
            "--network_args",
            "verbose=True",
            "network_reg_dims=.*self_attn.*=2",
            "network_reg_lrs=.*self_attn.*=9e-5",
            "--unet_self_attn_dim",
            "16",
            "--unet_self_attn_lr",
            "1e-4",
        ]
    )

    apply_anima_lora_component_args(args)

    assert args.network_args == [
        "verbose=True",
        r"network_reg_dims=.*self_attn.*=2,blocks\.\d+\.self_attn\..*=16",
        r"network_reg_lrs=.*self_attn.*=9e-5,blocks\.\d+\.self_attn\..*=0.0001",
    ]
