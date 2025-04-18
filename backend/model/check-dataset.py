import sys
from datasets import load_dataset

import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.utils.preprocess import preprocess_dataset


###### DATA SOURCES ######

# https://huggingface.co/datasets/tasksource/logical-fallacy
# dataset = load_dataset("tasksource/logical-fallacy")  


# https://huggingface.co/datasets/MidhunKanadan/logical-fallacy-classification
dataset = load_dataset("MidhunKanadan/logical-fallacy-classification")


###### PREPROCESSING ######
tokenized_dataset = preprocess_dataset(dataset, input_key="statement")


###### PRINT #######
# print(dataset)
print(dataset["train"][0])
