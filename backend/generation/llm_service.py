import os
import time

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")


client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.8-flash"


def generate_answer(question, evidence):

    context_parts = []

    for item in evidence:
        context_parts.append(
            f"""
Document: {item['document']}
Page: {item['page']}
Chunk ID: {item['chunk_id']}

Evidence:
{item['text']}
"""
        )

    context = "\n-----------------------------\n".join(
        context_parts
    )

    prompt = f"""
You are ResearchLens, an evidence-grounded research assistant.

Answer the user's question using ONLY the research evidence
provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the evidence is insufficient, explicitly say:
   "The retrieved evidence does not provide sufficient information
   to answer this question."
4. Give a concise but informative answer.
5. Preserve important technical terminology.
6. Cite every important claim using this format:
   [Document, Page X]
7. Do not create citations for information that is not present
   in the evidence.

USER QUESTION:
{question}

RETRIEVED RESEARCH EVIDENCE:
{context}
"""

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            return response.text

        except Exception as error:

            if attempt == max_attempts - 1:
                raise error

            wait_time = 2 ** attempt

            print(
                f"Gemini request failed. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)