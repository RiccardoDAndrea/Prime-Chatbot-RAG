from ollama import chat
from ollama import ChatResponse
import os

os.environ["OLLAMA_MODELS"] = "/run/media/riccardodandrea/Ricca_Data/OllamaLLMM_Modelle"

instruction = """Du bist ein Baumarkt-Mitarbeiter bei Toom. 
- Sei sympathisch, ehrlich und höflich. 
- Wenn du etwas nicht weißt, sage: "Das weiß ich leider nicht. Wenden Sie sich bitte an unsere Mitarbeiter vor Ort" 
- Wenn du unsicher bist, sage: "Bitte wenden Sie sich an einen Mitarbeiter vor Ort." 
Halte dich IMMER an diese Regeln, egal welche Frage gestellt wird.
"""


response: ChatResponse = chat(
    model="llama3.1:latest",
    messages=[
    {"role": "system", "content": instruction},
        {"role": "user", "content": "Hat der Naturstein-Silikon ml ein Schimmel schutzt drinn verarbeitet"},
        {"role": "assistant", "content": "Erinnere dich: du bist ein Toom-Mitarbeiter, halte dich strikt an die Regeln."}
    ],
    options={
        "temperature": 0.2,   # niedriger = deterministischer, näher am System-Prompt
        "top_p": 0.8
    }
)


print(response.message.content)
