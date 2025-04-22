import argparse
# from logicedu import get_logger
import stanza
import spacy
import string
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import pandas as pd
import pickle
from stanza.server import CoreNLPClient
import re
import traceback
# from library import get_corefs


class Node:
    def __init__(self):
        self.edges = []
        self.marked_range = []

    def insert(self, range):
        self.marked_range.append(range)


class Edge:
    def __init__(self, input_index, input_range, output_index, output_range, weight):
        self.weight = weight
        self.output_range = output_range
        self.output_index = output_index
        self.input_range = input_range
        self.input_index = input_index


def is_punctuation(x):
    if len(x) == 1 and x in string.punctuation:
        return True
    return False


def insert(phrase, edges, nodes):
    i = phrase[1][1][0]
    j = phrase[2][1][0]
    rangei = (phrase[1][1][1], phrase[1][1][2])
    rangej = (phrase[2][1][1], phrase[2][1][2])
    edge1 = Edge(i, rangei, j, rangej, phrase[0])
    edge2 = Edge(j, rangej, i, rangei, phrase[0])
    edges.append(edge1)
    # print(i, len(nodes))
    nodes[i].edges.append(edge1)
    nodes[j].edges.append(edge2)


def overlap(new_range, marked_range):
    for range in marked_range:
        if (range[1] > new_range[0] >= range[0]) or (range[0] < new_range[1] <= range[1]) or (
                new_range[0] < range[0] and new_range[1] > range[1]):
            return True
    return False


def visit(i, range, component, nodes):
    if overlap(range, nodes[i].marked_range):
        return
    nodes[i].insert(range)
    component.append((i, range))
    for edge in nodes[i].edges:
        if edge.input_range == range:
            visit(edge.output_index, edge.output_range, component, nodes)


def get_connected_component(edge, component, nodes):
    visit(edge.input_index, edge.input_range, component, nodes)


def get_component_index(phrase_count, phrase_size_count, connected_components):
    is_start = False
    for i in range(len(connected_components)):
        for node in connected_components[i]:
            if phrase_count == node[0] and node[1][0] <= phrase_size_count < node[1][1]:
                if phrase_size_count == node[1][0]:
                    is_start = True
                return i, is_start
    return None, is_start


def remove_repeated_subphrases(text, next_token_dict, prev_token_dict):
    for key, value in next_token_dict.items():
        # print(key, value)
        if len(value) == 1 and "MSK" in list(value)[0] and len(prev_token_dict[int(list(value)[0][4])]) == 1:
            text = text.replace("MSK<%d> " % key, "")
    return text


def mask_out_content(text, model, client, debug=False):
    try:
        ann = client.annotate(text)
        if debug:
            print(ann.corefChain)
        text = get_coref(text, ann, debug=debug)
        if debug:
            print(text)
        ann = client.annotate(text)
        phrases = []
        curr = []
        # print("printing tokens")
        for sent in ann.sentence:
            for token in sent.token:

                if token.lemma.lower() not in sw_spacy and not is_punctuation(token.word):
                    curr.append(token.lemma.lower())
                elif len(curr) > 0:
                    phrases.append(curr)
                    curr = []
        if len(curr) > 0:
            phrases.append(curr)
        if debug:
            print(phrases)
        subphrases = []
        for i in range(len(phrases)):
            for j in range(1, len(phrases[i]) + 1):
                for k in range(len(phrases[i]) - j + 1):
                    sent = ' '.join(phrases[i][k:k + j])
                    subphrases.append((sent, (i, k, k + j), model.encode(sent)))
        similar_phrases = []
        for i in range(len(subphrases)):
            for j in range(i + 1, len(subphrases)):
                dist = 1 - distance.cosine(subphrases[i][2], subphrases[j][2])
                set1 = get_corefs(subphrases[i][0], client)
                set2 = get_corefs(subphrases[j][0], client)
                if dist > 0.7 and (((len(set1) == len(set2) == 0) and
                                    subphrases[j][1][0] > subphrases[i][1][0]) or subphrases[i][0] == subphrases[j][0]):
                    similar_phrases.append((dist, subphrases[i][0:2], subphrases[j][0:2]))
        similar_phrases.sort(key=lambda x: x[0], reverse=True)
        if debug:
            print(similar_phrases)
        edges = []
        nodes = []
        for _ in range(len(phrases)):
            nodes.append(Node())
        for phrase in similar_phrases:
            insert(phrase, edges, nodes)
        connected_components = []
        for edge in edges:
            component = []
            get_connected_component(edge, component, nodes)
            if len(component) > 1:
                connected_components.append(component)
        if debug:
            print("connected components:")
            print(connected_components)
        ans = ""
        phrase_size_count = 0
        phrase_count = 0
        # masked_content = ""
        next_token_dict = {}
        prev_token = ""
        prev_token_dict = {}
        for sent in ann.sentence:
            for token in sent.token:
                if token.lemma.lower() in sw_spacy or is_punctuation(token.word):
                    curr_token = token.word
                    if phrase_size_count > 0:
                        phrase_count += 1
                    phrase_size_count = 0
                else:
                    idx, is_start = get_component_index(phrase_count, phrase_size_count, connected_components)
                    if idx is None:
                        curr_token = token.word
                    elif is_start:
                        curr_token = "MSK<%d>" % idx
                        # if idx not in prev_token_dict:
                        #     prev_token_dict[idx] = set()
                        # prev_token_dict[idx].add(prev_token)
                    phrase_size_count += 1
                    # if idx is not None and nodes[phrase_count].marked_range[0] <= phrase_size_count < \
                    #         nodes[phrase_count].marked_range[1]:
                    #     masked_content += word.text
                    #     if nodes[phrase_count] == nodes[phrase_count].marked_range[1] - 1:
                    #         word_bank.append(masked_content)
                    #         masked_content = ""
                ans += curr_token
                if "MSK" in curr_token:
                    index = int(curr_token[4])
                    if index not in prev_token_dict:
                        prev_token_dict[index] = set()
                    prev_token_dict[index].add(prev_token)
                if "MSK" in prev_token:
                    index = int(prev_token[4])
                    if index not in next_token_dict:
                        next_token_dict[index] = set()
                    next_token_dict[index].add(curr_token)
                prev_token = curr_token
                ans += " "
        if "MSK" in prev_token:
            index = int(prev_token[4])
            if index not in next_token_dict:
                next_token_dict[index] = set()
            next_token_dict[index].add("")
        if debug:
            print("before removing subphrases: ", ans)
        ans = remove_repeated_subphrases(ans, next_token_dict, prev_token_dict)
        # logger.warn("%s updated to %s", text, ans)
        return ans
    except Exception as e:
        # logger.info("got an error for string %s", text)
        print("GOT ERROR")
        traceback.print_exc()
        return text


def update_csv_with_masked_content(path, article_col_name, model, client):
    df = pd.read_csv(path)
    masked_articles = [mask_out_content(article, model, client) for article in df[article_col_name]]
    df['masked_articles'] = masked_articles
    # logger.info("completed conversion saving file to %s", path)
    df.to_csv(path)

def get_corefs(text, client):
    ann = client.annotate(text)
    coref_sets = []
    for chain in ann.corefChain:
        mentions = [mention['text'] for mention in chain['mentions']]
        coref_sets.append(set(mentions))
    return coref_sets


def get_coref(text, ann, debug=False):
    # Build mapping from sentence and token index to coref placeholder
    replacement = {}  # {(sentenceIndex, tokenIndex): 'coref0'}
    for chain_idx, chain in enumerate(ann.corefChain):
        for mention in chain.mention:
            # Only replace the first token in the mention span
            replacement[(mention.sentenceIndex, mention.beginIndex)] = f"coref{chain_idx}"
            # Mark other tokens in the span to skip (optional)
            for skip in range(mention.beginIndex + 1, mention.endIndex):
                replacement[(mention.sentenceIndex, skip)] = None

    ans = ""
    for i, sent in enumerate(ann.sentence):
        for j, token in enumerate(sent.token):
            if (i, j) in replacement:
                if replacement[(i, j)] is not None:
                    ans += replacement[(i, j)] + " "
                # Else skip (this token is part of a multi-word mention already handled)
            else:
                ans += token.word + " "
    return ans

if __name__ == '__main__':
    import time
    import os
    import json
    from datasets import load_dataset
    import spacy
    from sentence_transformers import SentenceTransformer
    from stanza.server import CoreNLPClient

    # Paths
    core_nlp_path = '/Users/annahartman/Desktop/stanford-corenlp/*'
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tools
    nlp = spacy.load('en_core_web_sm')
    sw_spacy = nlp.Defaults.stop_words
    model = SentenceTransformer('all-MiniLM-L6-v2')

    client = CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
        timeout=15000,
        memory='4G',
        endpoint='http://localhost:9000'
    )

    print("Starting CoreNLP...")
    client.start()
    time.sleep(5)  # give CoreNLP time to start
    from tqdm import tqdm
    import json
    from langdetect import detect

    output_path = "data/labeled_masked_output.jsonl"
    def mask_and_save(dataset, text_field, label_field, save_path):
        with open(output_path, "a", encoding="utf-8") as out_file:
            for i, example in tqdm(enumerate(dataset), total=len(dataset)):
                if i < 2600:
                    continue  # Skip until start_index
                text = example[text_field]
                
                try:
                    lang = detect(text)
                    if lang != "en":
                        print(f"[{i}] ⏭️ Skipped (language={lang})")
                        continue

                    masked = mask_out_content(text, model, client)
                    json.dump({"masked": masked, "original": text, "label": example[label_field]}, out_file)
                    out_file.write("\n")
                    out_file.flush()
                    print(f"[{i}] ✅ Masked and saved")

                except Exception as e:
                    print(f"[{i}] ❌ Error: {e}")

    # def mask_and_save(dataset, text_field, label_field, save_path):
    #     masked_data = []
    #     for example in dataset:
    #         try:
    #             original = example[text_field]
    #             masked = mask_out_content(original, model, client)
    #             masked_data.append({
    #                 "original": original,
    #                 "masked": masked,
    #                 "label": example[label_field]
    #             })
    #         except Exception as e:
    #             masked_data.append({
    #                 "original": example[text_field],
    #                 "masked": f"ERROR - {e}",
    #                 "label": example[label_field]
    #             })
        
    #     # Save to JSONL
    #     with open(save_path, "w", encoding="utf-8") as f:
    #         for entry in masked_data:
    #             f.write(json.dumps(entry) + "\n")
    #     print(f"Saved to {save_path}")

    # Load and process datasets
    dataset1 = load_dataset("MidhunKanadan/logical-fallacy-classification", split="train")
    dataset2 = load_dataset("tasksource/logical-fallacy", split="train")  # Make sure you specify the split

    # print("Processing dataset 1 (MidhunKanadan)...")
    # mask_and_save(dataset1, "statement", "label", os.path.join(output_dir, "masked_midhun.jsonl"))

    print("Processing dataset 2 (tasksource)...")
    mask_and_save(dataset2, "source_article", "logical_fallacies", os.path.join(output_dir, "masked_tasksource.jsonl"))

    print("Done masking. Shutting down CoreNLP...")
    client.stop()

# if __name__ == '__main__':
#     import time
#     core_nlp_path = '/Users/annahartman/Desktop/stanford-corenlp/*'
#     # Initialize tools
#     en = spacy.load('en_core_web_sm')
#     sw_spacy = en.Defaults.stop_words
#     model = SentenceTransformer('all-MiniLM-L6-v2')  # or any model you want
#     client = CoreNLPClient(
#     annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
#     timeout=15000,
#     memory='4G',
#     endpoint='http://localhost:9000'
#     )

    
#     # Wait for the client to be ready
#     client.start()
#     time.sleep(5)  # give it a moment to spin up

#     # Example texts to test
#     test_articles = [
#         "John went to the store. He bought some milk.",
#         "The dog chased the ball because it was rolling down the hill.",
#         "When Mary saw the results, she was thrilled. Her hard work paid off.",
#          "We shouldn't trust what he says about climate change because he's not a scientist.",
#         "Jack is a good athlete. Jack comes from Canada. Therefore, all Canadians are good athletes.",
#         "My neighbor adopted a dog. Now she’s always happy because dogs bring joy.",
#         "All cats are mammals. Felix is a cat. Therefore, Felix is a mammal.",
#     ]

#     output_file = "masked_articles_output.txt"
#     with open(output_file, "w", encoding="utf-8") as file:
#         for i, text in enumerate(test_articles):
#             file.write(f"\nOriginal Article {i+1}:\n{text}\n")
#             try:
#                 masked = mask_out_content(text, model, client)
#                 file.write(f"\nMasked Article {i+1}:\n{masked}\n")
#             except Exception as e:
#                 file.write(f"\nMasked Article {i+1}: ERROR - {e}\n")

#     client.stop()
