"""Dedicated component-level CLI arguments for Anima LoRA training."""

import argparse
import logging


logger = logging.getLogger(__name__)


ANIMA_LORA_COMPONENT_PATTERNS = (
    ("unet_self_attn", r"blocks\.\d+\.self_attn\..*"),
    ("unet_cross_attn", r"blocks\.\d+\.cross_attn\..*"),
    ("unet_mlp", r"blocks\.\d+\.mlp\..*"),
    ("te_self_attn", r"layers\.\d+\.self_attn\..*"),
    ("te_cross_attn", r"layers\.\d+\.cross_attn\..*"),
    ("te_mlp", r"layers\.\d+\.mlp\..*"),
)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def add_anima_lora_component_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Anima LoRA component rank and learning rate")
    labels = {
        "unet_self_attn": "DiT self-attention",
        "unet_cross_attn": "DiT cross-attention",
        "unet_mlp": "DiT MLP",
        "te_self_attn": "Qwen3 text-encoder self-attention",
        "te_cross_attn": "Qwen3 text-encoder cross-attention",
        "te_mlp": "Qwen3 text-encoder MLP",
    }
    for option_prefix, _ in ANIMA_LORA_COMPONENT_PATTERNS:
        label = labels[option_prefix]
        group.add_argument(
            f"--{option_prefix}_dim",
            type=_non_negative_int,
            default=None,
            help=f"LoRA rank (dimension) for {label}; unset uses --network_dim, and 0 disables the component",
        )
        group.add_argument(
            f"--{option_prefix}_lr",
            type=_non_negative_float,
            default=None,
            help=f"LoRA learning rate for {label}; unset uses the applicable component or global learning rate",
        )


def _merge_generated_network_arg(network_args: list[str], key: str, generated_pairs: list[str]) -> list[str]:
    if not generated_pairs:
        return network_args

    # train_network converts network_args to a dict, so the last duplicate key is
    # the effective value. Preserve that behavior, then put the user-provided
    # regexes first because lora_anima uses the first matching regex.
    prefix = f"{key}="
    existing_value = None
    remaining_args = []
    for network_arg in network_args:
        if network_arg.startswith(prefix):
            existing_value = network_arg[len(prefix) :]
        else:
            remaining_args.append(network_arg)

    values = []
    if existing_value:
        values.append(existing_value)
    values.extend(generated_pairs)
    remaining_args.append(f"{key}={','.join(values)}")
    return remaining_args


def apply_anima_lora_component_args(args: argparse.Namespace) -> None:
    """Translate the dedicated component options to lora_anima regex settings."""
    arg_values = vars(args)
    dim_pairs = []
    lr_pairs = []
    for option_prefix, pattern in ANIMA_LORA_COMPONENT_PATTERNS:
        dim = arg_values.get(f"{option_prefix}_dim")
        if dim is not None:
            dim_pairs.append(f"{pattern}={dim}")

        lr = arg_values.get(f"{option_prefix}_lr")
        if lr is not None:
            lr_pairs.append(f"{pattern}={lr}")

    network_args = list(arg_values.get("network_args") or [])
    network_args = _merge_generated_network_arg(network_args, "network_reg_dims", dim_pairs)
    network_args = _merge_generated_network_arg(network_args, "network_reg_lrs", lr_pairs)
    args.network_args = network_args or None

    if arg_values.get("te_cross_attn_dim") is not None or arg_values.get("te_cross_attn_lr") is not None:
        logger.warning(
            "Qwen3 currently has no cross-attention modules, so --te_cross_attn_dim/--te_cross_attn_lr "
            "will not match a module in the current Anima text encoder."
        )
