import ollama
from ollama import chat
from ollama import ChatResponse

# List of available models
ollamaModel_LLMlist = ["llama3.1", "llama3.2"]

def initalise_memory(len_history=int):
    messages = []
    MAX_HISTORY = len_history  # Begrenze die Historie auf 10 Nachrichten
    return messages, MAX_HISTORY

def llmwithHistory(prompt:str, messages, MAX_HISTORY):
    try:
        if prompt.lower() in ["exit", "quit", "bye"]:
            print("Chat beendet. Bis zum nächsten Mal! 👋")
            return False  # Beende die Schleife

        # Füge den neuen Nutzerinput zur Nachrichten-Historie hinzu
        messages.append({"role": "user", "content": prompt})

        # KI-Antwort generieren (Simulation durch Funktion `chat`)
        response = chat('llama3.2', messages=messages)

        # Antwort der KI zur Historie hinzufügen
        messages.append({"role": "assistant", "content": response.message.content})

        # Historie begrenzen (alte Nachrichten entfernen)
        if len(messages) > MAX_HISTORY:
            messages = messages[-MAX_HISTORY:]

        print(f"Assistant: {response.message.content}\n")
        return True

    except KeyboardInterrupt:
        print("\nChat unterbrochen. Bis bald! 👋")
        return False

messages, MAX_HISTORY = initalise_memory(len_history=10)

while True:
    user_input = input("You: ")
    if not llmwithHistory(prompt=user_input, messages=messages, MAX_HISTORY=MAX_HISTORY):
        break