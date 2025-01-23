import ollama
from ollama import chat
from ollama import ChatResponse


ollamaModel_LLMlist = ["llama3.1", "llama3.2"]



def ollamaLLM_response(prompt:str, model:int):
    """
    prompt:
    ----
    User str input to ask a question

    model:int
    ---
    Choose from a list the model
    
    """
    response: ChatResponse = chat(model=ollamaModel_LLMlist[int], messages=[
    {
        'role': 'user',
        'content': f'{prompt}',
    },
    ])
    answer = response['message']['content']
    # or access fields directly from the response object
    return answer

ollamaModelvision_list = ["llava","llama3.2-vision"]
