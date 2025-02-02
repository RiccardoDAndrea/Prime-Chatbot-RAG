import ollama
from ollama import chat
from ollama import ChatResponse

# List of available models
ollamaModel_LLMlist = ["llama3.1", "llama3.2"]


def ollamaLLM_response(prompt: str, model: str) -> str:
    """
    Generate a response using a selected LLM model.

    Parameters:
    -----------
    prompt : str
        User input as a question or instruction.

    Returns:
    --------
    str
        The response from the model.

    """
    # Get the model name

    # Chat with the selected model
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""
                You are a helpful Assistant to every question.
                {prompt}
                """,
            },
        ],
    )

    # Extract the response content
    answer = response.message['content']
    return answer

print(ollamaLLM_response(prompt="What can i do for you", model="llama3.1"))