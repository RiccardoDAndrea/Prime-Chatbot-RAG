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


messages = []
MAX_HISTORY = 10  # Begrenze die Historie auf 10 Nachrichten

while True:
    try:
        user_input = input("You: ")  

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chat beendet. Bis zum nächsten Mal! 👋")
            break  # Beende die Schleife

        # Füge den neuen Nutzerinput zur Nachrichten-Historie hinzu
        messages.append({"role": "user", "content": user_input})

        # KI-Antwort generieren (Simulation durch Funktion `chat`)
        response = chat('llama3.2', messages=messages)

        # Antwort der KI zur Historie hinzufügen
        messages.append({"role": "assistant", "content": response.message.content})

        # Historie begrenzen (alte Nachrichten entfernen)
        if len(messages) > MAX_HISTORY:
            messages = messages[-MAX_HISTORY:]

        print(f"Assistant: {response.message.content}\n")

    except KeyboardInterrupt:
        print("\nChat unterbrochen. Bis bald! 👋")
        break



















