from fastmcp import Client
from google import genai
import asyncio
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

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
                filtered down to LAT, LON, and brightness temps 1 and 2 data from 
                January 1, 2021 and summarize the data for me. Compare the 
                brightness temp datas.
                """,
            config=genai.types.GenerateContentConfig(
                # Reduce randomness in response
                temperature=0,
                # Give Gemini access to the MCP server tools
                tools=[mcp_client.session],
            ),
        )

        # Display Gemini's output
        print(response.text)

        # Uncomment plotting call if asking for full DataFrame in JSON format
        # NOTE: Getting full DataFrame in JSON is difficult with token limits
        # plot_json_data(response.text)


# Function to plot data from JSON format
def plot_json_data(json_data: str):
    """Plot JSON data representing a pandas DataFrame using Matplotlib.

    Args:
        json_data (str): A string of JSON data in records orientation representing data to plot.
    """
    # Convert Gemini's JSON output into a DataFrame
    df = pd.read_json(StringIO(json_data), orient="records")

    # Plot the data obtained from the query
    plot_col = df.columns[2]
    plt.figure(figsize=(12, 8))
    plt.scatter(df["LON"], df["LAT"], s=15, c=df[plot_col])
    plt.title(f"AMSU Brightness Temperature for {plot_col}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
