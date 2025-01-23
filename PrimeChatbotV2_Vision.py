import ollama
from ollama import chat
from ollama import ChatResponse



models = ollama.list()["models"]
ollamaModelvision_list = [model["model"] for model in models] # return all downloades ollama models

user_input = "What is strange about this picture"

def ollama_vision(prompt:str, model:int):
    """
    TODO: Desc
    
    """
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
    model = response['model']    
    answer = response['message']['content']
    return model, answer

print(ollama_vision(prompt=user_input, model=1))
