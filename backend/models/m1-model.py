from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import json
import os

# Define the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("MidhunKanadan/logical-fallacy-classification", split="train")
texts = dataset["statement"]
labels = dataset["label"]

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
output_dir = "saved_models/m1_model"
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/label2id.json", "w") as f:
    json.dump(label2id, f)
with open(f"{output_dir}/id2label.json", "w") as f:
    json.dump(id2label, f)

# Train/test split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, encoded_labels, test_size=0.2)

# Tokenizer (using BERT tokenizer)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Dataset class for text encoding
class FallacyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "label": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# DataLoaders
train_dataset = FallacyDataset(train_texts, train_labels, tokenizer)
val_dataset = FallacyDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Compute class weights for balanced classification
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define the BERT-based model for classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
model.to(device)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training loop with early stopping and gradient clipping
num_epochs = 5
patience = 2
best_val_acc = 0
epochs_without_improvement = 0


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()

        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.logits, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Accuracy = {correct / total:.4f}")

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f"Validation Accuracy = {val_accuracy:.4f}")

    # Early stopping
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        epochs_without_improvement = 0
        # Save model
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break

    # Update learning rate scheduler
    scheduler.step()

print(f"Best Validation Accuracy = {best_val_acc:.4f}")

# Save model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
