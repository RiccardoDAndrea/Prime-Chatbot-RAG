# import ollama
# from ollama import chat
# from ollama import ChatResponse



# models = ollama.list()["models"]

# ollamaModelvision_list = [model["model"] for model in models] # return all downloades ollama models
# print(ollamaModelvision_list)
# user_input = "What is strange about this picture"

# def ollama_vision(prompt:str, model:int):
#     """
#     prompt:
#     ---
#     Is the user question for the Chatbot 

#     model:
#     ---
#     int for the Ollama models
   
#     """
#     with open('/run/media/riccardodandrea/Ricca_Data/', 'rb') as file:
#         response = ollama(
#             model= ollamaModelvision_list[model] ,
#             messages=[
#             {
#                 'role': 'user',
#                 'content': f'{user_input}',
#                 'images': [file.read()],
#             },
#             ],
#         )
#     model = response['model']    
#     answer = response['message']['content']
#     return model, answer

# #print(ollama_vision(prompt=user_input, model=1))
