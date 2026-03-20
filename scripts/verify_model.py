import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL")

print(f"--- FinSight Model Verification ---")
print(f"Model ID: {model}")

if not api_key:
    print("Error: ANTHROPIC_API_KEY not found in environment.")
    exit(1)

client = Anthropic(api_key=api_key)

print("Sending test message...")
try:
    message = client.messages.create(
        model=model,
        max_tokens=20,
        messages=[{"role": "user", "content": "Respond with 'Connectivity OK' if you can read this."}]
    )
    print(f"Success! Response from {model}:")
    print(f" >> {message.content[0].text}")
except Exception as e:
    print(f"Verification Failed!")
    print(f"Error details: {e}")
