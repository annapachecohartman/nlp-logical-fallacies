import spacy
from sentence_transformers import SentenceTransformer, util
import re

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

def split_premise_hypothesis(text):
    """
    Naively split text into premise and hypothesis using discourse markers.
    """
    markers = ["because", "since", "so", "therefore", "thus", "as a result", "hence"]
    for marker in markers:
        parts = text.lower().split(marker)
        if len(parts) > 1:
            pre = text[:text.lower().find(marker) + len(marker)].strip()
            hypo = text[text.lower().find(marker) + len(marker):].strip()
            return pre, hypo
    return text.strip(), text.strip()

def get_candidate_spans(doc):
    """
    Extract candidate spans: noun chunks + named entities.
    """
    spans = set()
    for chunk in doc.noun_chunks:
        spans.add(chunk.text.strip())
    for ent in doc.ents:
        spans.add(ent.text.strip())
    return list(spans)

def paraphrase_mask(premise, hypothesis, threshold=0.6, debug=False):
    """
    Replace semantically similar spans (including multi-word ones) between premise and hypothesis 
    with shared [MSK#] placeholders using sentence embeddings and lemmatization.
    """

    doc1 = nlp(premise)
    doc2 = nlp(hypothesis)

    def extract_spans(doc):
        spans = {}
        for chunk in list(doc.noun_chunks) + list(doc.ents):
            # Clean and lemmatize
            lemma = " ".join([token.lemma_.lower() for token in chunk if not token.is_stop and token.is_alpha])
            original = chunk.text.strip()
            if lemma and len(lemma.split()) <= 6:  # Max 6-word span
                spans[lemma] = original
        return spans

    spans1 = extract_spans(doc1)
    spans2 = extract_spans(doc2)

    all_lemmas = list(set(spans1.keys()) | set(spans2.keys()))
    if debug:
        print(f"\n Candidate Spans: {list(set(spans1.values()) | set(spans2.values()))}")

    embeddings = model.encode(all_lemmas, convert_to_tensor=True)
    groups = []
    used = set()

    for i, lemma_i in enumerate(all_lemmas):
        if lemma_i in used:
            continue
        group = [lemma_i]
        for j, lemma_j in enumerate(all_lemmas):
            if i != j and lemma_j not in used:
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim > threshold or lemma_i == lemma_j + "s" or lemma_i + "s" == lemma_j:
                    group.append(lemma_j)
                    used.add(lemma_j)
        used.add(lemma_i)
        if len(group) > 1:
            groups.append(group)

    if debug:
        print(f"\n Matched Groups:")
        for i, g in enumerate(groups, 1):
            print(f"  MSK{i}: {[spans1.get(s, spans2.get(s, s)) for s in g]}")

    def mask_text(text, groups, span_map):
        masked = text
        for i, group in enumerate(groups, 1):
            token = f"[MSK{i}]"
            for lemma in group:
                orig_phrase = span_map.get(lemma)
                if orig_phrase:
                    # Use word boundaries and ignore case, prioritize whole phrases
                    pattern = re.compile(r'\b' + re.escape(orig_phrase) + r'\b', flags=re.IGNORECASE)
                    masked = pattern.sub(token, masked)
        return masked

    masked_prem = mask_text(premise, groups, spans1)
    masked_hypo = mask_text(hypothesis, groups, spans2)

    return masked_prem, masked_hypo


def preprocess_argument(text, structure_aware=True, debug=False):
    premise, hypothesis = split_premise_hypothesis(text)

    if not structure_aware:
        return premise, hypothesis

    masked_prem, masked_hypo = paraphrase_mask(premise, hypothesis, debug=debug)
    return masked_prem, masked_hypo



### EXAMPLES

examples = [
    "We shouldn't trust what he says about climate change because he's not a scientist.",
    "Jack is a good athlete. Jack comes from Canada. Therefore, all Canadians are good athletes.",
    "My neighbor adopted a dog. Now she’s always happy because dogs bring joy.",
    "All cats are mammals. Felix is a cat. Therefore, Felix is a mammal.",
]

for i, ex in enumerate(examples):
    print(f"\n--- Example {i + 1} ---")
    prem, hypo = preprocess_argument(ex, structure_aware=False)
    print("Original Premise:", prem)
    print("Original Hypothesis:", hypo)

    masked_prem, masked_hypo = preprocess_argument(ex, structure_aware=True, debug=True)
    print("\nStructure-Aware Premise:", masked_prem)
    print("Structure-Aware Hypothesis:", masked_hypo)

