import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model + tokenizer
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "backend/saved_models/m2_model"
DEVICE = "cpu"  # safer on macOS

with open(f"{ADAPTER_PATH}/id2label.json", "r") as f:
    id2label = json.load(f)

# Convert string keys to integers if needed
id2label = {int(k): v for k, v in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.to(DEVICE)
model.eval()

import re

def predict_fallacy_m2(text: str) -> str:
    prompt = f"<|user|>\nAnalyze the following argument and identify any logical fallacies:\n\n{text}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to find a known label name in the output
    for label in id2label.values():
        if label.lower() in result.lower():
            return label
    
    return "Unknown fallacy"



# EXAMPLE USAGE
examples = [
    "If we legalize marijuana, then everyone will start using heroin.",
    "You're just saying that because you're a Democrat.",
    "Water freezes at 0 degrees Celsius.",
    "My opponent says we should tax the rich, so they're saying we should abolish all taxes",
    "You're either with America or against America."
]

for text in examples:
    label = predict_fallacy_m2(text)
    print(f"\"{text}\"\n→ Predicted: {label}\n")