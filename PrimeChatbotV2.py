## Prime-Chatbot Version 2 
### Die neue Version so die fahigkeit haben  mehre Model zu beinhalten
### Dabei soll auch eine Optimerite Rag Kompatibilitat hinzugefugt werden

import ollama

from ollama import chat
from ollama import ChatResponse
ollamaModel_LLMlist = ["llama3.1", "llama3.2"]



def ollamaLLM_response(prompt:str, model:int):
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




user_input = "What type of tank is one picture?"
def ollama_vision(prompt:str, model:int):
    with open('/run/media/riccardodandrea/Ricca_Data/ArmoredEye/Main_battle_tanks/Russia/T_90/T_90_40.jpg', 'rb') as file:
        response = ollama.chat(
            model= ollamaModelvision_list[model] ,
            messages=[
            {
                'role': 'user',
                'content': f'{user_input}',
                'images': [file.read()],
            },
            ],
        )
    answer = response['message']['content']
    return answer

#print(response["model"] + " " + (response["created_at"]))
#print(response["message"]["content"])