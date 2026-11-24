"""Task 2. Pass (release sample, decoded turn records) pairs and a CONCH encoder."""
from .metrics import stream_scores


def evaluate(cases, encode):
    streams = []
    for sample, predictions in cases:
        references = sample["reference_descriptions"]
        actions = ["silent" if text == "silent" else "update" for text in references]
        turns = {t["turn_index"]: t for t in predictions}
        streams.append((actions, references, turns))
    return stream_scores(streams, encode)
