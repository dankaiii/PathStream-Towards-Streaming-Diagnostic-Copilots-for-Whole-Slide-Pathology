"""Task 1. Pass (release sample, prediction records) pairs to evaluate."""
from .metrics import decision, gate_scores


def evaluate(cases):
    pairs = []
    for sample, predictions in cases:
        turns = {turn["turn_index"]: turn for turn in predictions}
        for i, action in enumerate(sample["reference_actions"][1:], 1):
            pairs.append((action, decision(turns.get(i))))
    return gate_scores(pairs)
