import ollama
from ollama import chat
from ollama import ChatResponse

# List of available models
ollamaModel_LLMlist = ["llama3.1", "llama3.2"]

def ollamaLLM_response(prompt: str, model: int) -> str:
    """
    Generate a response using a selected LLM model.

    Parameters:
    -----------
    prompt : str
        User input as a question or instruction.
    model : int
        Index of the model to use from ollamaModel_LLMlist.

    Returns:
    --------
    str
        The response from the model.

    """
    # Get the model name
    selected_model = ollamaModel_LLMlist[model]

    # Chat with the selected model
    response: ChatResponse = chat(
        model=selected_model,
        messages=[
            {
                "role": "user",
                "content": f"""
                You are a helpful Assistant that starts every answer with a funny Joke.
                {prompt}
                """,
            },
        ],
    )

    # Extract the response content
    answer = response.message['content']
    return answer


# Example usage
print(ollamaLLM_response(prompt="come estas?", model=0))
