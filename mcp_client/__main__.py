import asyncio
import logging
import sys

from mcp_client.cli import parse_args
from mcp_client.mcp_client import MCPClient


def configure_logging(verbose: bool) -> None:
    """Send logs to stderr, surface mcp_client debug logs only when verbose."""
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("mcp_client").setLevel(
        logging.DEBUG if verbose else logging.WARNING
    )


async def main():
    """Run the MCP client with the specified options."""
    # Parse CLI arguments to determine what operation to perform
    args = parse_args()

    configure_logging(args.verbose)

    try:
        # Initialize and connect to the MCP server
        async with MCPClient(args.target) as client:
            # Execute the requested action
            if args.members:
                await client.list_all_members()
            elif args.chat:
                await client.run_chat(model=args.model)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    asyncio.run(main())


def cli_main():
    """Entry point for the mcp-client CLI app."""
    asyncio.run(main())
