from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch


BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "checkpoints/fallacy-qlora"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)


print("Configuring QLoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)


print("Loading dataset...")
dataset = load_dataset("logical_fallacy")  # Replace with your own if needed

def format_example(example):
    text = f"Analyze the following argument and identify any logical fallacies:\n\n{example['text']}\n\nFallacies:"
    return {"input_ids": tokenizer(text, truncation=True, padding="max_length", max_length=512)["input_ids"]}

print("Tokenizing...")
tokenized_dataset = dataset["train"].map(format_example)
tokenized_dataset.set_format(type="torch", columns=["input_ids"])


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=20,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="no",
    logging_dir="./logs",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()


print(f"Saving adapter to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
