import json
import torch
from recommendation import *
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
base_dir = pathlib.Path(__name__).parent.absolute()
recommend_job = 0
recommend_candidate = 0

with open("intent.json", "r") as f:
    data = json.load(f)
intents = {}
for intent in data["intents"]:
    name = intent["name"].lower()
    patterns = [pattern.lower() for pattern in intent["patterns"]]
    responses = intent["responses"]
    intents[name] = {"patterns": patterns, "responses": responses}

def get_response(text):
    # Encode user input using tokenizer
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    # Generate response using model
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    # Decode response using tokenizer
    chat_history = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    # Remove user input from chat history
    chat_history = chat_history.replace(text, "")
    # Remove any leading or trailing whitespace
    chat_history = chat_history.strip()
    return chat_history

