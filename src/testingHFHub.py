import os
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from . import config

api_token = config.HUGGINGFACEHUB_API_TOKEN
model_name = "google/flan-t5-large"
test_prompt = "What is the capital of France?"

if not api_token:
    print("Error: HUGGINGFACEHUB_API_TOKEN environment variable not set.")
else:
    try:
        print(f"Attempting direct API call to {model_name} with task 'text-generation'...")
        client = InferenceClient(model=model_name, token=api_token)

        # Use the specific helper if available (preferred)
        # response = client.text2text_generation(test_prompt)

        # Or use the generic post method explicitly setting the task
        response = client.post(json={"inputs": test_prompt}, task="text-generation")

        print("Direct API call successful:")
        print(response) # Adjust based on what post returns (might be bytes)

    except HfHubHTTPError as e:
        print(f"Direct API call failed with HTTP error: {e}")
        print(f"Response content: {e.response.content}") # See if HF API gives more info
    except ValueError as e:
         print(f"Direct API call failed with ValueError: {e}") # Check if same error occurs
    except Exception as e:
        print(f"Direct API call failed with unexpected error: {e}")

