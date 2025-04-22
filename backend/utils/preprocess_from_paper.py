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
    from datasets import load_dataset, Dataset
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
    def mask_and_save(dataset, text_field, label_field):
        with open(output_path, "a", encoding="utf-8") as out_file:
            for i, example in tqdm(enumerate(dataset), total=len(dataset)):
                print(example)
                if i < 0:
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
    none_dataset = Dataset.from_list([{"text": sample["statement"], "label": sample["label"]} for sample in new_samples])

    print("Processing none dataset...")
    mask_and_save(none_dataset, "text", "label")
    # print("Processing dataset 1 (MidhunKanadan)...")
    # mask_and_save(dataset1, "statement", "label", os.path.join(output_dir, "masked_midhun.jsonl"))

    # print("Processing dataset 2 (tasksource)...")
    # mask_and_save(dataset2, "source_article", "logical_fallacies")

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
