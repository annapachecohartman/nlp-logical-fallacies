import json
import os
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
import torch
import random

# Load tokenizer and model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("MidhunKanadan/logical-fallacy-classification", split="train")



# Add custom "None" fallacy samples
new_samples = [
    {"statement": "The Earth revolves around the Sun.", "label": "None"},
    {"statement": "Water boils at 100 degrees Celsius at sea level.", "label": "None"},
    {"statement": "There are seven continents on Earth.", "label": "None"},
    {"statement": "The square root of 9 is 3.", "label": "None"},
    {"statement": "Dogs are domesticated animals.", "label": "None"},
    {"statement": "The human body contains 206 bones.", "label": "None"},
    {"statement": "Photosynthesis occurs in the chloroplasts of plant cells.", "label": "None"},
    {"statement": "Mount Everest is the highest mountain above sea level.", "label": "None"},
    {"statement": "The Pacific Ocean is the largest ocean on Earth.", "label": "None"},
    {"statement": "The freezing point of water is 0°C.", "label": "None"},
    {"statement": "Light travels faster than sound.", "label": "None"},
    {"statement": "The moon orbits the Earth.", "label": "None"},
    {"statement": "Gravity pulls objects toward the Earth’s center.", "label": "None"},
    {"statement": "A triangle has three sides.", "label": "None"},
    {"statement": "Penguins are birds that cannot fly.", "label": "None"},
    {"statement": "Carbon dioxide is a greenhouse gas.", "label": "None"},
    {"statement": "Whales are mammals, not fish.", "label": "None"},
    {"statement": "The Great Wall of China is visible from space only under specific conditions.", "label": "None"},
    {"statement": "Venus is the hottest planet in the solar system.", "label": "None"},
    {"statement": "A leap year occurs every four years.", "label": "None"},
    {"statement": "Sound requires a medium to travel.", "label": "None"},
    {"statement": "The human heart has four chambers.", "label": "None"},
    {"statement": "Mars is known as the Red Planet.", "label": "None"},
    {"statement": "A healthy diet includes a variety of nutrients.", "label": "None"},
    {"statement": "The Nile is the longest river in Africa.", "label": "None"},
    {"statement": "The chemical formula for water is H₂O.", "label": "None"},
    {"statement": "The brain is the control center of the nervous system.", "label": "None"},
    {"statement": "Electricity is conducted by metals like copper.", "label": "None"},
    {"statement": "Rainbows are caused by light refraction in water droplets.", "label": "None"},
    {"statement": "A solar eclipse occurs when the Moon passes between Earth and the Sun.", "label": "None"},
    {"statement": "Cows are herbivores.", "label": "None"},
    {"statement": "Saturn has rings made of ice and rock.", "label": "None"},
    {"statement": "The internet was developed in the late 20th century.", "label": "None"},
    {"statement": "Recycling reduces landfill waste.", "label": "None"},
    {"statement": "Oxygen is essential for human respiration.", "label": "None"},
    {"statement": "Bees are important pollinators.", "label": "None"},
    {"statement": "Helium is lighter than air.", "label": "None"},
    {"statement": "Bacteria can be both harmful and beneficial.", "label": "None"},
    {"statement": "The Pythagorean theorem applies to right triangles.", "label": "None"},
    {"statement": "Fossils provide evidence of past life.", "label": "None"},
    {"statement": "The seasons are caused by the tilt of the Earth's axis.", "label": "None"},
    {"statement": "A compass points to magnetic north.", "label": "None"},
    {"statement": "Volcanoes form at tectonic plate boundaries.", "label": "None"},
    {"statement": "The lungs are responsible for gas exchange.", "label": "None"},
    {"statement": "The Eiffel Tower is in Paris, France.", "label": "None"},
    {"statement": "Atoms are the basic units of matter.", "label": "None"},
    {"statement": "DNA carries genetic information.", "label": "None"},
    {"statement": "Clouds are formed from condensed water vapor.", "label": "None"},
    {"statement": "Lightning is caused by a discharge of static electricity.", "label": "None"},
    {"statement": "Sharks have cartilaginous skeletons.", "label": "None"},
    {"statement": "The pancreas regulates blood sugar levels.", "label": "None"},
    {"statement": "Antarctica is the coldest continent.", "label": "None"},
    {"statement": "Mammals give birth to live young.", "label": "None"},
    {"statement": "Gravity affects all objects with mass.", "label": "None"},
    {"statement": "Birds have hollow bones to aid in flight.", "label": "None"},
    {"statement": "Humans have 23 pairs of chromosomes.", "label": "None"},
    {"statement": "The speed of light is approximately 299,792 km/s.", "label": "None"},
    {"statement": "Caterpillars turn into butterflies through metamorphosis.", "label": "None"},
    {"statement": "Zebras are native to Africa.", "label": "None"},
    {"statement": "Water is composed of two hydrogen atoms and one oxygen atom.", "label": "None"},
    {"statement": "Airplanes fly due to lift generated by wings.", "label": "None"},
    {"statement": "Electric current is measured in amperes.", "label": "None"},
    {"statement": "The skin is the body’s largest organ.", "label": "None"},
    {"statement": "Reptiles are cold-blooded animals.", "label": "None"},
    {"statement": "The Amazon is the largest rainforest in the world.", "label": "None"},
    {"statement": "The moon has phases due to its orbit around Earth.", "label": "None"},
    {"statement": "Temperature is measured in Celsius or Fahrenheit.", "label": "None"},
    {"statement": "The liver helps detoxify substances in the body.", "label": "None"},
    {"statement": "An octopus has eight arms.", "label": "None"},
    {"statement": "Bananas are a good source of potassium.", "label": "None"},
    {"statement": "The piano is a stringed instrument.", "label": "None"},
    {"statement": "Earthquakes are caused by movement of tectonic plates.", "label": "None"},
    {"statement": "Copper is a good conductor of electricity.", "label": "None"},
    {"statement": "Hydrogen is the lightest element.", "label": "None"},
    {"statement": "Ice melts at 0°C.", "label": "None"},
    {"statement": "Mercury is the closest planet to the Sun.", "label": "None"},
    {"statement": "Trees produce oxygen through photosynthesis.", "label": "None"},
    {"statement": "An adult human has 32 teeth.", "label": "None"},
    {"statement": "Snakes shed their skin regularly.", "label": "None"},
    {"statement": "The sun is a star at the center of our solar system.", "label": "None"},
    {"statement": "A liter is equal to 1000 milliliters.", "label": "None"},
    {"statement": "Most plants need sunlight to grow.", "label": "None"},
    {"statement": "Gold is a soft, yellow metal.", "label": "None"},
    {"statement": "The Sahara is the largest hot desert in the world.", "label": "None"},
    {"statement": "Tides are caused by the gravitational pull of the moon.", "label": "None"},
    {"statement": "Frogs undergo metamorphosis from tadpoles.", "label": "None"},
    {"statement": "The spine protects the spinal cord.", "label": "None"},
    {"statement": "A circle has 360 degrees.", "label": "None"},
    {"statement": "Sound is measured in decibels.", "label": "None"},
    {"statement": "Plastic can take hundreds of years to decompose.", "label": "None"},
    {"statement": "Water covers about 71% of the Earth’s surface.", "label": "None"},
    {"statement": "Lions are apex predators in their ecosystem.", "label": "None"},
    {"statement": "Mathematics is the study of numbers and patterns.", "label": "None"},
    {"statement": "The keyboard is an input device for a computer.", "label": "None"},
    {"statement": "A rainbow has seven colors.", "label": "None"},
    {"statement": "Cacti store water in their stems.", "label": "None"},
    {"statement": "Camels can survive without water for long periods.", "label": "None"},
    {"statement": "The Earth orbits the Sun.", "label": "None"},
    {"statement": "Water freezes at 0 degrees Celsius.", "label": "None"},
    {"statement": "Exercise is good for your health.", "label": "None"},
    {"statement": "Eating fruits and vegetables can improve your diet.", "label": "None"},
    {"statement": "The capital of France is Paris.", "label": "None"},
    {"statement": "Photosynthesis is the process by which plants make food.", "label": "None"},
    {"statement": "The human body contains 206 bones.", "label": "None"},
    {"statement": "Batteries store electrical energy.", "label": "None"},
    {"statement": "Airplanes are used for long-distance travel.", "label": "None"},
    {"statement": "Vaccines help protect against diseases.", "label": "None"},
    {"statement": "The Pacific Ocean is the largest ocean on Earth.", "label": "None"},
    {"statement": "Mount Everest is the tallest mountain above sea level.", "label": "None"},
    {"statement": "A healthy breakfast can boost your energy levels.", "label": "None"},
    {"statement": "The Moon affects the tides of Earth.", "label": "None"},
    {"statement": "Recycling helps reduce environmental waste.", "label": "None"},
    {"statement": "The internet allows people to communicate globally.", "label": "None"},
    {"statement": "Sound travels slower than light.", "label": "None"},
    {"statement": "Drinking enough water helps maintain hydration.", "label": "None"},
    {"statement": "Learning a new language can improve cognitive skills.", "label": "None"},
    {"statement": "The boiling point of water is 100 degrees Celsius at sea level.", "label": "None"},
    {"statement": "Solar panels convert sunlight into electricity.", "label": "None"},
    {"statement": "Wearing a seatbelt can reduce injury in a car crash.", "label": "None"},
    {"statement": "The Sahara is the largest hot desert in the world.", "label": "None"},
    {"statement": "Reading regularly can expand your vocabulary.", "label": "None"},
    {"statement": "The heart pumps blood throughout the body.", "label": "None"},
]


# Add to existing dataset
for sample in new_samples:
    dataset = dataset.add_item(sample)

texts = dataset["statement"]
labels = dataset["label"]

# Map string labels to integers
unique_labels = sorted(set(labels))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
encoded_labels = [label2id[label] for label in labels]
num_classes = len(unique_labels)

# Save label mappings
output_dir = "saved_models/m2_model"
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/label2id.json", "w") as f:
    json.dump(label2id, f)
with open(f"{output_dir}/id2label.json", "w") as f:
    json.dump(id2label, f)

# Format dataset in instruction-tuning format (ChatML / Alpaca-style)
def format_example(example):
    instruction = "Analyze the following argument and identify any logical fallacies:"
    input_text = example["statement"]
    label_id = label2id[example["label"]]  # map label to its corresponding number
    output_text = f"Fallacy ID: {label_id}"
    return {
        "text": f"<|user|>\n{instruction}\n\n{input_text}\n<|assistant|>\n{output_text}",
        "label": label_id  # include numerical label if training model on IDs
    }

# Combine, shuffle, and format
formatted_dataset = dataset.map(format_example)

# Tokenize
def tokenize_sample(sample):
    return tokenizer(sample["statement"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = formatted_dataset.map(tokenize_sample, batched=True)

# Split into train/validation sets
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# Load base model and apply QLoRA
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    output_dir="./tinyllama-fallacy",
    save_total_limit=2,
    fp16=False,
    report_to="none"
)

# Proof of concept
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=2,
#     num_train_epochs=1,
#     max_steps=20,  # Fast test run
#     logging_steps=5,
#     save_steps=20,
#     eval_steps=20,
#     save_total_limit=1,
#     report_to="none",
# )


# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Train and save
trainer.train()
trainer.save_model("./tinyllama-fallacy/final")
tokenizer.save_pretrained("./tinyllama-fallacy/final")





# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from peft import PeftModel
# import torch

# BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
# ADAPTER_PATH = "checkpoints/fallacy-qlora"  # Path to QLoRA fine-tuned adapter

# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# print("Loading base model...")
# model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto"  
# )

# print("Applying QLoRA adapter...")
# model = PeftModel.from_pretrained(model, ADAPTER_PATH)
# model.eval()


# def predict_fallacies(text: str) -> dict:
#     prompt = f"""Analyze the following argument and identify any logical fallacies:\n\n{text}\n\nFallacies:"""

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=100,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True
#         )

#     generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     result_text = generated.split("Fallacies:")[-1].strip()

#     return {"fallacies": result_text}
