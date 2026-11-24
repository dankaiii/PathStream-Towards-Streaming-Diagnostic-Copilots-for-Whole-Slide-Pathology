"""Task 4: build four-choice questions from the ordered views in a clip."""
from bisect import bisect_right
from itertools import accumulate, combinations, product
import random


SEMANTIC_PROMPT = """Use only the ordered view descriptions. Extract up to four
subject/attribute groups, each with two to four distinct states. Prefer two
appearance attributes of the same subject and attributes comparable across views.
For every view and state, use supported, contradicted, or unknown. A known
relation requires a verbatim quote from that view with the same subject, scope
and degree. Treat unmentioned and ambiguous findings as unknown. The order is
browsing order. Return groups with judgments in view order, following this schema:
{"groups": [{"subject": "...", "attribute": "...", "kind": "appearance",
 "states": ["state A", "state B"], "judgments": [
 {"state A": {"relation": "supported", "quote": "..."},
  "state B": {"relation": "unknown", "quote": ""}}]}]}.
Use kind "appearance" for morphology and "spatial" for spatial attributes."""


def states(group, view, relation):
    return [state for state in group["states"]
            if group["judgments"][view][state]["relation"] == relation]


def contrasts(group, view):
    return product(states(group, view, "supported"), states(group, view, "contradicted"))


def candidates(groups, middle, end):
    """Each option assigns one state to each (label, attribute group, view) column."""
    endpoints = [("beginning", 0), ("end", end)] if end else [("beginning", 0)]
    appearance = [group for group in groups if group["kind"] == "appearance"]
    for when, view in endpoints:
        for first, second in combinations(appearance, 2):
            if first["subject"] != second["subject"] or first["attribute"] == second["attribute"]:
                continue
            question = f"At the {when} of the video, which combination describes the {first['subject']}?"
            columns = [(g["attribute"], g, view) for g in (first, second)]
            for a, b in product(contrasts(first, view), contrasts(second, view)):
                yield "content_recognition", question, columns, product(a, b)

    if not end:
        return
    for group in groups:
        feature = f"the {group['attribute']} of the {group['subject']}"
        columns = [("Beginning", group, 0), ("end", group, end)]
        for a, b in product(states(group, 0, "supported"), states(group, end, "supported")):
            if a != b:
                question = f"From the beginning to the end of the video, which pair describes {feature}?"
                yield "temporal_change", question, columns, product((a, b), repeat=2)

        if middle in (0, end):
            continue
        columns = [("Beginning", group, 0), ("middle", group, middle), ("end", group, end)]
        for sequence in product(*(states(group, v, "supported") for _, _, v in columns)):
            if len(set(sequence)) != 3:
                continue
            a, b, c = sequence
            options = [sequence, (b, a, c), (c, b, a), (a, c, b)]
            question = f"Which sequence describes {feature} at the beginning, middle, and end of the video?"
            yield "temporal_order", question, columns, options


def make_question(kind, question, columns, choices, descriptions, rng):
    """Keep one fully supported option and three explicitly contradicted alternatives."""
    options = []
    for choice in choices:
        parts, claims = [], []
        for (label, group, view), state in zip(columns, choice):
            judgment = group["judgments"][view][state]
            relation, quote = judgment["relation"], judgment["quote"]
            if relation not in ("supported", "contradicted") or not quote or quote not in descriptions[view]:
                return None
            parts.append(f"{label}: {state}")
            claims.append({"view": view, "relation": relation, "quote": quote})
        supported = all(claim["relation"] == "supported" for claim in claims)
        options.append(("; ".join(parts) + ".", supported, claims))

    if sum(supported for _, supported, _ in options) != 1:
        return None
    rng.shuffle(options)
    letters = dict(zip("ABCD", options))
    return {
        "task": 4, "question_type": kind, "question": question,
        "options": {letter: text for letter, (text, _, _) in letters.items()},
        "instruction": "Answer with exactly one letter: A, B, C, or D.",
        "answer": next(letter for letter, (_, supported, _) in letters.items() if supported),
        "option_evidence": {letter: claims for letter, (_, _, claims) in letters.items()},
    }


def build(clip, annotate, rng=random):
    """annotate(prompt, payload) returns parsed JSON. The middle anchor uses frame count."""
    views = clip["views"]
    descriptions = [view["description"] for view in views]
    annotation = annotate(SEMANTIC_PROMPT, {"descriptions": descriptions})
    ends = list(accumulate(view["frame_count"] for view in views))
    middle = bisect_right(ends, ends[-1] // 2)
    for kind, question, columns, choices in candidates(annotation["groups"], middle, len(views) - 1):
        sample = make_question(kind, question, columns, choices, descriptions, rng)
        if sample:
            yield sample
