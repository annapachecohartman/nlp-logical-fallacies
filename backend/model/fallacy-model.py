from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
ADAPTER_PATH = "checkpoints/fallacy-qlora"  # Path to QLoRA fine-tuned adapter

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"  
)

print("Applying QLoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()


def predict_fallacies(text: str) -> dict:
    prompt = f"""Analyze the following argument and identify any logical fallacies:\n\n{text}\n\nFallacies:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result_text = generated.split("Fallacies:")[-1].strip()

    return {"fallacies": result_text}
