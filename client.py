from fastmcp import Client
from google import genai
import asyncio
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import gzip
import base64

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
            model="gemini-2.0-flash",
            # Example query to show that the client works
            contents="""List the available datasets. Then, access the first dataset 
                filtered down to LAT and LON data and BRITCSTC.TMBR_00001 from 
                January 1, 2021 and return the data as JSON so I can decompress 
                it and make a DataFrame. Only output the compressed data, with 
                no extra filler text and no ```json ``` wrapper text.
                """,
            config=genai.types.GenerateContentConfig(
                # No randomness in response
                temperature=0,
                # Give Gemini access to the MCP server tools
                tools=[mcp_client.session],
            ),
        )

        # Display Gemini's output
        print(response.text)

        # Decompress the returned data
        if response.text:
            # TODO: FIX THIS LINE NOT WORKING: Incorrect padding
            compressed_bytes = base64.b64decode(response.text.encode("utf-8"))
        else:
            raise Exception("No response was returned by Gemini.")
        decompressed_data = gzip.decompress(compressed_bytes)
        # Convert Gemini's output into a DataFrame
        df = pd.DataFrame(StringIO(decompressed_data.decode("utf-8")))

        # Plot the data obtained from the query
        plot_col = "BRITCSTC.TMBR_00001"
        plt.figure(figsize=(12, 8))
        plt.scatter(df["LON"], df["LAT"], s=15, c=df[plot_col])
        plt.title(f"AMSU Brightness Temperature for {plot_col}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
