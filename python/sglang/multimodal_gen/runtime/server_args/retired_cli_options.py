# SPDX-License-Identifier: Apache-2.0
"""CLI spellings from 0.5.x that diffusion no longer accepts.

Each entry maps a retired option to the message that names its replacement.
``ServerArgs.from_cli_args`` checks the arguments argparse did not recognize
against this table before the dynamic ``--<component>-path`` alias can take
them, and before the generic "unrecognized arguments" error, so a 0.5.x launch
command fails with the new spelling instead of a bare parse error.
"""

_REMOVED_DISAGG_OPTION = (
    "was removed: it had no effect. Drop it; the other --disagg-* options are "
    "unchanged."
)

RETIRED_CLI_OPTIONS: dict[str, str] = {
    # The generic --<component>-path alias would otherwise store it as a
    # component path named "mask_strategy_file" and ignore it.
    "--mask-strategy-file-path": (
        "was removed. Pass the sliding tile attention mask strategy as "
        "--attention-backend-config mask_strategy_file_path={value} "
        "(with --attention-backend sliding_tile_attn)."
    ),
    "--disagg-downstream-wait-timeout": _REMOVED_DISAGG_OPTION,
    "--disagg-instance-id": _REMOVED_DISAGG_OPTION,
    "--disagg-max-slots-per-instance": _REMOVED_DISAGG_OPTION,
    "--disagg-role-device": _REMOVED_DISAGG_OPTION,
    "--disagg-transfer-backend": _REMOVED_DISAGG_OPTION,
    "--disagg-transfer-pin-memory": _REMOVED_DISAGG_OPTION,
    "--disagg-transfer-redundancy": _REMOVED_DISAGG_OPTION,
    "--sp-attention-mode": (
        "was removed. Choose the sequence-parallel exchange by its degree: "
        "--kv-gather-degree N for kv_gather, --ulysses-degree N for ulysses, "
        "where N is the SP degree."
    ),
    # Only reaches this table where it is not an option: `sglang serve`.
    "--adjust-frames": (
        "is a per-request sampling option, not a server option. Send "
        "adjust_frames in the request, or pass --adjust-frames to "
        "`sglang generate`. Models that need it off, such as MOVA, already "
        "default it to false."
    ),
    "--num-layers": (
        "is not a CLI option. For Ming-Image layer decomposition, put "
        "'Number of layers: N' in --prompt (4 when the prompt names none), "
        "or send num_layers in an HTTP request."
    ),
    "--output-size": "was removed. Use --width W --height H instead.",
}


def _option_value(unknown_args: list[str], index: int) -> str:
    arg = unknown_args[index]
    if "=" in arg:
        return arg.split("=", 1)[1]
    if index + 1 < len(unknown_args) and not unknown_args[index + 1].startswith("-"):
        return unknown_args[index + 1]
    return "<path>"


def reject_retired_cli_options(unknown_args: list[str]) -> None:
    """Raise SystemExit naming the replacement for the first retired option."""
    for index, arg in enumerate(unknown_args):
        if not arg.startswith("--"):
            continue
        option = "--" + arg[2:].split("=", 1)[0].replace("_", "-")
        message = RETIRED_CLI_OPTIONS.get(option)
        if message is not None:
            message = message.format(value=_option_value(unknown_args, index))
            raise SystemExit(f"error: {option} {message}")
