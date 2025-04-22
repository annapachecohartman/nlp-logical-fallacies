import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForCausalLM
import json
from peft import PeftModel


model_path = "backend/saved_models/m1_model"

# Load label mappings
with open(f"{model_path}/id2label.json", "r") as f:
    id2label = json.load(f)

# Convert string keys to integers if needed
id2label = {int(k): v for k, v in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_fallacy_m1(text):
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # Decode predicted class
    predicted_label = id2label[predicted_class_id]
    return predicted_label


# EXAMPLE USAGE
examples = [
    "If we legalize marijuana, then everyone will start using heroin.",
    "You're just saying that because you're a Democrat.",
    "Water freezes at 0 degrees Celsius.",
    "My opponent says we should tax the rich, so they're saying we should abolish all taxes",
    "You're either with America or against America."
]

for text in examples:
    label = predict_fallacy_m1(text)
    print(f"\"{text}\"\n→ Predicted: {label}\n")
