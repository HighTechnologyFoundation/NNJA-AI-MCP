import argparse


def parse_args():
    """Parse command line arguments and return parsed args."""
    parser = argparse.ArgumentParser(description="A minimal MCP client")

    # Optional positional argument for the server script / URL
    parser.add_argument(
        "target",
        nargs="?",
        default="server.py",
        help=(
            "path to the MCP server script (stdio) "
            "or an http(s):// URL of a running server "
            "(default: server.py)"
        ),
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

    # Optional flag for overriding the Gemini model
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Gemini model for --chat "
            "(overrides the GEMINI_MODEL env var; default: gemini-3.1-flash-lite)"
        ),
    )

    # Optional flag for verbose (debug) mode
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging (to stderr)",
    )

    return parser.parse_args()
