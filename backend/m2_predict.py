import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

# Load the Hugging Face model and tokenizer
MODEL_NAME = "mithrandir22/tinyLLama-Logical-Fallacy"
DEVICE = "cpu"  # Change to "cuda" if you're on a GPU-enabled system

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# Optional: Known fallacy label list (can be extracted from your dataset or output patterns)
known_labels = [
    "Slippery Slope", "Appeal to worse problems", "Appeal to nature", "False dilemma",
    "None", "Appeal to authority", "Hasty generalization", "Appeal to majority",
    "Hasty Generalization", "Appeal to tradition"
]

SYSTEM_PROMPT = """You are a logic expert. For each statement, identify exactly one logical fallacy from the list below that best matches the reasoning error. If there is no fallacy, reply with "none".


FALLACIES:
- Appeal to nature: Judgment is based solely on whether the subject of judgment is “natural” or “unnatural.”
- Appeal to worse problems: Dismissing an argument or complaint due to what are perceived to be more important problems
- False dilemma: Presenting only two options when more exist
- Hasty generalization: Drawing broad conclusions from a small sample
- Slippery slope: Claiming one action will inevitably lead to extreme negative consequences
- Appeal to authority: Using an expert of dubious credentials or using only one opinion to promote a product or idea.
- Appeal to majority: Claiming something is true/right because many people believe it
- Appeal to tradition: Arguing something is good/right because it's traditional or has been done for a long time

EXAMPLES:
Text: Senator Randall isn't lying when she says she cares about her constituents—she is a senator so she wouldn't lie to people she cares about.
Fallacy: Appeal to authority

Text: I love eating burgers.
Fallacy: none

Text: If we ban Hummers because they are bad for the environment, eventually the government will ban all cars, so we should not ban Hummers.
Fallacy: False Dilemma
"""


def predict_fallacy_m2(text: str) -> str:
    prompt = SYSTEM_PROMPT + f"\nText: {text}\nFallacy:"
    # inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    pipe = pipeline("text-generation", model=MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=5,
    #         do_sample=True,
    #         temperature=0.1,
    #         top_k=20,
    #         top_p=0.8,
            
    #     )

    # result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    llm_output = pipe(prompt, max_new_tokens=5, do_sample=True, temperature=0.1, top_k=20, top_p=0.8)[0]["generated_text"]
    # print(llm_output)
    pairs = {}
    blocks = re.split(r'\n(?=Text:)', llm_output)
    for block in blocks:
        text_match = re.search(r'Text:\s*(.+)', block)
        fallacy_match = re.search(r'Fallacy:\s*(.+)', block)
        if text_match and fallacy_match:
            text = text_match.group(1).strip()
            fallacy = fallacy_match.group(1).strip()
            pairs[text] = fallacy
    t = pairs.get(text, "Fallacy not found")

    return t
    # print(llm_output)

    # print(result)
    # # Extract prediction based on known labels
    # for label in known_labels:
    #     if label.lower() in result.lower():
    #         return label

    return "Unknown fallacy"

# def predict_fallacy_m2(text: str) -> str:
#     prompt = SYSTEM_PROMPT + f"\nText: {text}\nFallacy:"
    
#     pipe = pipeline("text-generation", model=MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

#     llm_output = pipe(prompt, max_new_tokens=20, do_sample=True, temperature=0.1, top_k=20, top_p=0.8)[0]["generated_text"]

#     # Extract the fallacy using regex
#     fallacy_match = re.search(r'Fallacy:\s*(.*)', llm_output)
#     if fallacy_match:
#         prediction = fallacy_match.group(1).strip()
#         for label in known_labels:
#             if label.lower() in prediction.lower():
#                 return label
#         return prediction

#     return "Unknown fallacy"


# EXAMPLE USAGE
examples = [
    "If we legalize marijuana, then everyone will start using heroin.",
    "You're just saying that because you're a Democrat.",
    "Water freezes at 0 degrees Celsius.",
    "My opponent says we should tax the rich, so they're saying we should abolish all taxes",
    "You're either with America or against America."
]

# for text in examples:
#     label = predict_fallacy_m2(text)
#     print(f"\"{text}\"\n→ Predicted: {label}\n")
