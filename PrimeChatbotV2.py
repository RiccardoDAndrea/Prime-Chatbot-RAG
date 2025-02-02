## Prime-Chatbot Version 2 
### Die neue Version so die fahigkeit haben  mehre Model zu beinhalten
### Dabei soll auch eine Optimerite Rag Kompatibilitat hinzugefugt werden

import ollama

from ollama import chat
from ollama import ChatResponse
ollamaModel_LLMlist = ["llama3.1", "llama3.2"]



def ollamaLLM_response(prompt:str, model:int):
    """
    TODO: Desc
    
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


#print(response["model"] + " " + (response["created_at"]))
#print(response["message"]["content"])
