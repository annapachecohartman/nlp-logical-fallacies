from datasets import DatasetDict
from transformers import AutoTokenizer

# Initialize the tokenizer (you can parametrize this if needed)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def preprocess_function(example, input_key="source_article"):
    """
    Formats and tokenizes input text for logical fallacy detection.

    Args:
        example (dict): A dictionary containing the input example.
        input_key (str): The key in the dataset to use as the source text.

    Returns:
        dict: Tokenized input for the model.
    """
    prompt = f"Analyze the following argument and identify any logical fallacies:\n\n{example[input_key]}\n\nFallacies:"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

def preprocess_dataset(dataset: DatasetDict, input_key="source_article", batched=False):
    """
    Applies preprocessing to all splits of the dataset.

    Args:
        dataset (DatasetDict): The full Hugging Face dataset object with splits.
        input_key (str): The key to use as the input text (e.g. "source_article" or "statement").
        batched (bool): Whether to use batched processing.

    Returns:
        DatasetDict: Tokenized dataset.
    """
    return dataset.map(lambda x: preprocess_function(x, input_key), batched=batched)
