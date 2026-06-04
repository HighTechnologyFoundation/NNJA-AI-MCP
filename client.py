import asyncio
import os

from dotenv import load_dotenv
from fastmcp import Client
from google import genai

# Load environment variables from .env file
load_dotenv()

# Get GEMINI_API_KEY environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize MCP and Gemini clients
mcp_client = Client("http://localhost:8000/mcp")
gemini_client = genai.Client(api_key=gemini_api_key)


async def main():
    async with mcp_client:
        print(f"Connected: {mcp_client.is_connected()}")

        # Have Gemini generate content based on our query and the available tools
        response = await gemini_client.aio.models.generate_content(
            model="gemini-3.1-flash-lite",
            # Example query to show that the client works
            contents="""List the available datasets. Then, access the first dataset 
                filtered down to LAT, LON, and brightness temp 1, 2, 3, 4, and 5 data 
                from January 1, 2021 and summarize the data for me. Compare the 
                brightness temp datum and provide me their stats.
                """,
            config=genai.types.GenerateContentConfig(
                # Give Gemini access to the MCP server tools
                tools=[mcp_client.session],
            ),
        )

        # Display Gemini's output
        print(response.text)


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
