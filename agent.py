import os
import json
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# GPT-4.1 Client 
AI_ENDPOINT = "https://models.github.ai/inference"
AI_MODEL = "openai/gpt-4.1"
AI_TOKEN = os.getenv("AI_TOKEN")

ai_client = ChatCompletionsClient(
    endpoint=AI_ENDPOINT,
    credential=AzureKeyCredential(AI_TOKEN),
)

# Recycling Agent System Prompt
RECYCLING_SYSTEM_PROMPT = """You are an AI Recycling Classification Agent for the Yalla Khadra platform.

Your task is to analyze the waste prediction or waste description provided by the system and determine whether the waste is recyclable.

Rules:

1. Return ONLY valid JSON.
2. Do NOT return markdown.
3. Do NOT return explanations outside the JSON object.
4. The response must always contain exactly these two properties:
   * AIIsRecyclable (boolean)
   * AIExplanation (string)
5. AIIsRecyclable must be:
   * true if the waste can reasonably be recycled.
   * false if the waste is generally non-recyclable.
6. AIExplanation must:
   * Explain how the waste can be recycled or reused.
   * Be useful for educating users.
   * Never exceed 250 characters.
7. If confidence is low, choose the most likely classification based on common recycling practices.
8. Ensure the generated JSON is always syntactically valid.

Response Format:
{"AIIsRecyclable": true, "AIExplanation": "Can be recycled into new plastic products."}

Important:
Return ONLY the JSON object. No additional text, comments, code fences, or markdown."""


def get_recycling_classification(waste_description: str) -> dict:
    """Send the waste prediction to GPT-4.1 and return the recycling classification."""
    try:
        response = ai_client.complete(
            messages=[
                SystemMessage(RECYCLING_SYSTEM_PROMPT),
                UserMessage(waste_description),
            ],
            temperature=0.3,
            top_p=1.0,
            model=AI_MODEL,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return {
            "AIIsRecyclable": False,
            "AIExplanation": "Unable to determine recyclability at this time.",
        }
