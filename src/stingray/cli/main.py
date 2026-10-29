from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


COMMANDS = {
    ("sensors", "merge"): {
        "help": "Merge sensor data",
        "target": "stingray.cli.sensors:main",
    },
    ("images", "abundance"): {
        "help": "Compute image abundance",
        "target": "stingray.images.abundance:main",
    },
    ("images", "frame-timestamp"): {
        "help": "Build frame timestamp CSV",
        "target": "stingray.images.build_frame_timestamps:main",
    },
    ("images", "add-media"): {
        "help": "Add media metadata to merged sensor CSV",
        "target": "stingray.images.add_media:main",
    },
    ("ctd", "download"): {
        "help": "Download NES-LTER CTD cruise data",
        "target": "stingray.ctd.download:main",
    },
    ("ctd", "concat"): {
        "help": "Concatenate Sea-Bird CNV casts into one CSV",
        "target": "stingray.ctd.concat:main",
    },
}

GROUPS = {
    "sensors": {
        "aliases": ["sensor"],
        "help": "Process and merge Stingray sensor observations",
    },
    "images": {
        "aliases": ["image"],
        "help": "Build frame metadata, attach media, and compute abundance",
    },
    "ctd": {
        "aliases": [],
        "help": "Download and compile CTD reference data",
    },
}


def package_version() -> str:
    try:
        return version("stingraytools")
    except PackageNotFoundError:
        return "unknown"


def load_target(target: str) -> Callable[[list[str] | None], None]:
    module_name, func_name = target.split(":", 1)
    module = import_module(module_name)
    func = getattr(module, func_name)

    if not callable(func):
        raise TypeError(f"Target is not callable: {target}")

    return func


def run_target(target: str, argv: list[str]) -> None:
    func = load_target(target)
    func(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stingray",
        description="Stingray command-line tools",
        epilog="Run 'stingray <group> --help' to list that group's commands.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"stingray {package_version()}",
    )

    groups = parser.add_subparsers(
        dest="group",
        title="command groups",
        metavar="<group>",
        required=True,
    )

    group_parsers: dict[str, argparse._SubParsersAction] = {}

    for (group, command), spec in COMMANDS.items():
        if group not in group_parsers:
            group_spec = GROUPS[group]
            group_parser = groups.add_parser(
                group,
                aliases=group_spec["aliases"],
                help=group_spec["help"],
                description=group_spec["help"],
                epilog=f"Run 'stingray {group} <command> --help' for all command options.",
            )
            group_parsers[group] = group_parser.add_subparsers(
                dest="command",
                title="commands",
                metavar="<command>",
                required=True,
            )

        command_parser = group_parsers[group].add_parser(
            command,
            help=spec["help"],
            description=spec["help"],
            add_help=False,  # let the target command handle -h/--help
        )
        command_parser.set_defaults(target=spec["target"])

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, passthrough_args = parser.parse_known_args(argv)

    target = getattr(args, "target", None)
    if target is None:
        parser.error("No command selected")

    run_target(target, passthrough_args)


if __name__ == "__main__":
    main()
