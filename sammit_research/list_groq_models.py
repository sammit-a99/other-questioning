import os
from groq import Groq

# Initialize the Groq Client
os.environ["GROQ_API_KEY"] = "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"
client = Groq()

# Free tier rate limits for Groq (as of 2026)
GROQ_FREE_TIER_LIMITS = {
    "RPM": 30,  # Requests per minute
    "RPD": 14400,  # Requests per day
    "TPM": 6000,  # Tokens per minute (input + output combined)
}

# List all available models
print("Available Groq Models (Free Tier Rate Limits):")
print("=" * 50)
print(f"Free Tier Limits: {GROQ_FREE_TIER_LIMITS}")
print("=" * 50)

try:
    models = client.models.list()
    for model in models.data:
        print(f"Model ID: {model.id}")
        print(f"Created: {model.created}")
        print(f"Owned By: {model.owned_by}")
        print(f"Free Tier RPM: {GROQ_FREE_TIER_LIMITS['RPM']}")
        print(f"Free Tier RPD: {GROQ_FREE_TIER_LIMITS['RPD']}")
        print("-" * 50)
except Exception as e:
    print(f"Error listing models: {e}")
