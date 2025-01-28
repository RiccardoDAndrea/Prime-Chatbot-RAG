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
                You are a helpful Assistant to every question.
                {prompt}
                """,
            },
        ],
    )

    # Extract the response content
    answer = response.message['content']
    return answer
#print(ollamaLLM_response(prompt="do you rember my name?", model=0))


from ollama import chat

messages = [
    {
        "role": "user",
        "content": "Why is the sky blue",
    },
    {
        "role": "assistant",
        "content":"The sky is blue because of the way the Earths atmosphere scatters sunlight"
    },
    {
        "role":"user",
        "content": "what is the weather in Tokyo?",
    },
    {
        "role": "assistant",
        "content":"The weather in Tokoyo is typically warm and humid during the summer months, with temperatures often exceeding 30 30°C (86°F). The city experiences a rainy season from June to September, with heavy rainfall and occasional typhoons. Winter is mild, with temperatures rarely dropping below freezing. The city is known for its high-tech and vibrant culture, with many popular tourist attractions such as the Tokyo Tower, Senso-ji Temple, and the bustling Shibuya district.",
    },
    ]

# while True:
#   user_input = "string"
#   response = chat(
#     'llama3.2',
#     messages=messages
#     + [
#       {'role': 'user', 'content': user_input},
#     ],
#   )

#   # Add the response to the messages to maintain the history
#   messages += [
#     {'role': 'user', 'content': user_input},
#     {'role': 'assistant', 'content': response.message.content},
#   ]
#   print(response.message.content + '\n') 




















