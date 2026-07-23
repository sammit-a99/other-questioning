import os
from google import genai

# Initialize the Gemini Client
os.environ["GOOGLE_API_KEY"] = "AIzaSyDXt8vdf15AxVGnWaZnUuviGAVYoXDIpKA"
client = genai.Client()

# Free tier rate limits for Gemini (as of 2026)
GEMINI_FREE_TIER_LIMITS = {
    "flash": {"RPM": 15, "RPD": 1500},
    "flash-lite": {"RPM": 30, "RPD": 1500},
    "pro": {"RPM": 5, "RPD": 50},
    "gemma": {"RPM": 30, "RPD": 1500},
    "imagen": {"RPM": 10, "RPD": 500},
}

def get_model_limits(model_name):
    """Get rate limits based on model name pattern."""
    model_name_lower = model_name.lower()
    if "flash-lite" in model_name_lower:
        return GEMINI_FREE_TIER_LIMITS["flash-lite"]
    elif "flash" in model_name_lower:
        return GEMINI_FREE_TIER_LIMITS["flash"]
    elif "pro" in model_name_lower:
        return GEMINI_FREE_TIER_LIMITS["pro"]
    elif "gemma" in model_name_lower:
        return GEMINI_FREE_TIER_LIMITS["gemma"]
    elif "imagen" in model_name_lower:
        return GEMINI_FREE_TIER_LIMITS["imagen"]
    else:
        return {"RPM": "Unknown", "RPD": "Unknown"}

# List all available models
print("Available Gemini Models (Free Tier Rate Limits):")
print("=" * 50)
print("Free Tier Limits by Model Type:")
for model_type, limits in GEMINI_FREE_TIER_LIMITS.items():
    print(f"  {model_type}: RPM={limits['RPM']}, RPD={limits['RPD']}")
print("=" * 50)

try:
    models = client.models.list()
    for model in models:
        limits = get_model_limits(model.name)
        print(f"Model Name: {model.name}")
        print(f"Display Name: {model.display_name}")
        print(f"Description: {model.description}")
        print(f"Free Tier RPM: {limits['RPM']}")
        print(f"Free Tier RPD: {limits['RPD']}")
        print("-" * 50)
except Exception as e:
    print(f"Error listing models: {e}")
