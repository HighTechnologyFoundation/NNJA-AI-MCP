import asyncio

from mcp_client.cli import parse_args
from mcp_client.mcp_client import MCPClient


async def main():
    """Run the MCP client with the specified options."""
    # Parse CLI arguments to determine what operation to perform
    args = parse_args()

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
