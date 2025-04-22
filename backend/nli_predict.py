import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Load model and tokenizer
model_dir = "backend/saved_models/nli_masked_model"  # or "nli_masked_model"

# Load model + tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# List of all label names used during training
fallacy_labels = [
    "ad hominem", "ad populum", "appeal to emotion", "circular reasoning", "equivocation",
    "fallacy of credibility", "fallacy of extension", "fallacy of logic", "fallacy of relevance",
    "false causality", "false dilemma", "faulty generalization", "intentional", "None"
]

def predict_fallacy_nli(text: str) -> str:
    premises = [text] * len(fallacy_labels)
    hypotheses = [f"This is an example of {label}." for label in fallacy_labels]

    inputs = tokenizer(premises, hypotheses, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    # Determine correct entailment index
    entailment_idx = model.config.label2id.get("entailment", 2)
    entailment_scores = logits[:, entailment_idx]
    
    best_index = torch.argmax(entailment_scores).item()

    print("\nScores:")
    for label, score in zip(fallacy_labels, entailment_scores.tolist()):
        print(f"{label:>25} → {score:.4f}")
    
    return fallacy_labels[best_index]


# EXAMPLE USAGE
examples = [
    "If we legalize marijuana, then everyone will start using heroin.",
    "You're just saying that because you're a Democrat.",
    "Water freezes at 0 degrees Celsius.",
    "My opponent says we should tax the rich, so they're saying we should abolish all taxes",
    "You're either with America or against America."
]

for text in examples:
    label = predict_fallacy_nli(text)
    print(f"\"{text}\"\n→ Predicted: {label}\n")