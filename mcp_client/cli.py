import argparse
import pathlib


def parse_args():
    """Parse command line arguments and return parsed args."""
    parser = argparse.ArgumentParser(description="A minimal MCP client")

    # Optional positional argument for the server script
    parser.add_argument(
        "server_path",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("server.py"),
        help="path to the MCP server script (default: server.py)",
    )

    # Mutually exclusive group: user can choose either listing members or starting chat
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--members",
        action="store_true",
        help="list the MCP server's tools, prompts, and resources",
    )
    group.add_argument(
        "--chat",
        action="store_true",
        help="start an AI-powered chat with MCP server integration",
    )
    return parser.parse_args()
