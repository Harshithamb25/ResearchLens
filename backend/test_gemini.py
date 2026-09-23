import os

from google import genai
from dotenv import load_dotenv


# Load variables from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")


# Create Gemini client
client = genai.Client(api_key=api_key)


# Send a simple test request
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explain what semantic search is in two simple sentences."
)


print("\nGemini response:\n")
print(response.text)