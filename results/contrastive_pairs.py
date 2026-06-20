import random
from datasets import load_dataset
from typing import Dict, List

random.seed(42)


def build_truthfulqa_pairs(n: int = 200) -> List[Dict]:
    """Return N contrastive pairs from TruthfulQA generation split."""
    ds = load_dataset("truthful_qa", "generation", split="validation")
    pairs = []
    for row in ds:
        q = row["question"].strip()
        corrects = row["correct_answers"]
        incorrects = row["incorrect_answers"]
        if not corrects or not incorrects:
            continue
        pairs.append({
            "question": q,
            "correct": corrects[0].strip(),
            "incorrect": incorrects[0].strip(),
            "domain": row.get("category", "unknown"),
            "prompt_correct": f"Q: {q}\nA: {corrects[0].strip()}",
            "prompt_incorrect": f"Q: {q}\nA: {incorrects[0].strip()}",
        })
        if len(pairs) >= n:
            break
    return pairs


def build_creak_pairs(n: int = 200) -> List[Dict]:
    """Return N contrastive pairs from CREAK by matching true/false claims per entity."""
    ds = load_dataset("amydeng2000/CREAK", split="train")
    entity_map = {}
    for row in ds:
        entity = row.get("entity", "unknown")
        label = row.get("label", "").lower().strip()
        sentence = row.get("sentence", "").strip()
        if entity not in entity_map:
            entity_map[entity] = {"true": [], "false": []}
        if label in ["true", "false"] and sentence:
            entity_map[entity][label].append(sentence)

    pairs = []
    for entity, claims in entity_map.items():
        if claims["true"] and claims["false"]:
            pairs.append({
                "question": f"Claim about {entity}",
                "correct": claims["true"][0],
                "incorrect": claims["false"][0],
                "domain": "entity_facts",
                "prompt_correct": f"Claim: {claims['true'][0]}",
                "prompt_incorrect": f"Claim: {claims['false'][0]}",
            })
        if len(pairs) >= n:
            break
    return pairs


def tokenize_pairs(pairs: List[Dict], model, device: str = "cuda"):
    """Tokenize contrastive pair prompts for the given model."""
    tokenized = []
    for p in pairs:
        tc = model.to_tokens(p["prompt_correct"], prepend_bos=True)[:, :64].to(device)
        ti = model.to_tokens(p["prompt_incorrect"], prepend_bos=True)[:, :64].to(device)
        tokenized.append((tc, ti, p["domain"]))
    return tokenized
