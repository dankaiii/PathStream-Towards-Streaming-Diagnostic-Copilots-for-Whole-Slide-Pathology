"""Task 2: build frame-aligned morphology references from a clip in clips.py."""

INSTRUCTION = (
    "Observe the pathology views in their presented order. Briefly describe "
    "the complete morphology of each newly presented view in English. "
    "Describe the first frame. Remain silent while a similar view continues "
    "after your description. A finding may be described again in another view; "
    "do not restrict descriptions to facts absent from the entire history. "
    "Do not give a diagnosis."
)


def build(clip):
    descriptions, evidence = [], []
    for view in clip["views"]:
        descriptions.append(view["description"])
        evidence.append(view["label_evidence"])
        for _ in range(view["frame_count"] - 1):
            descriptions.append("silent")
            evidence.append({})
    return {
        "task": 2, "fps": 1, "system": clip["system"], "instruction": INSTRUCTION,
        "reference_descriptions": descriptions, "reference_evidence_by_label": evidence,
    }
