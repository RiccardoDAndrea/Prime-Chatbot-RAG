import ollama

ollamaModel_list = ["llama3.2-vision", "llama3.2-vision", "llama3.2"]

user_input = "What is strange about this image?"

with open('/run/media/riccardodandrea/Ricca_Data/ArmoredEye/Main_battle_tanks/Russia/T_90/T_90_40.jpg', 'rb') as file:
  response = ollama.chat(
    model= ollamaModel_list[1] ,
    messages=[
      {
        'role': 'user',
        'content': f'{user_input}',
        'images': [file.read()],
      },
    ],
  )

print(response["model"] + " " + (response["created_at"]))
print(response["message"]["content"])