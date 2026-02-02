import os
from dotenv import load_dotenv

from mcp import ClientSession
from google import genai

# Load environment variables from .env file
load_dotenv()


class GeminiQueryHandler:
    """Handle Gemini API interaction and MCP tool execution."""

    def __init__(self, client_sesssion: ClientSession):
        self.client_session = client_sesssion
        if not (api_key := os.getenv("GEMINI_API_KEY")):
            raise RuntimeError(
                "Error: GEMINI_API_KEY environment variable not set",
            )
        self.gemini = genai.Client(api_key=api_key)
        self.chat = self.gemini.aio.chats.create(
            model="gemini-2.5-flash",
            config=genai.types.GenerateContentConfig(
                temperature=0,
                tools=[self.client_session],
            ),
        )

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available MCP tools."""
        # Get initial model's response and decision on tool calls
        # messages = [{"role": "user", "content": query}]
        response = await self.chat.send_message(query)

        return "Assistant: " + str(response.text)
