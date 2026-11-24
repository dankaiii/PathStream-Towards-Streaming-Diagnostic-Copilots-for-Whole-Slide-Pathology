"""Build shared Task 2 and 4 clips from one streaming session at a time."""

LABELS = {
    "cell_morphology": "Cell shape, cytoplasm, nuclei, atypia and mitoses.",
    "tissue_architecture": "Arrangement of cells and structures such as glands or nests.",
    "tissue_components": "Tissue constituents, stroma and extracellular material.",
    "inflammation_and_injury": "Inflammatory cells, necrosis, hemorrhage and related changes.",
    "spatial_distribution": "Explicit locations, distributions and relationships.",
}

LABEL_PROMPT = """Assign all applicable labels using only the supplied view description.
For each label, quote its supporting passages verbatim, preserving negation, degree
and uncertainty. Return {"assignments": [{"label_id": "...", "evidence_spans": ["..."]}]}.
An empty list means that none of the labels applies."""


def views(session, annotate):
    """Each update starts a view. annotate(prompt, payload) returns parsed JSON."""
    actions = session["supervision"]["frame_targets"]
    starts = [i for i, action in enumerate(actions) if action == "update"]
    descriptions = (turn["content"] for turn in session["conversation"]
                    if turn["role"] == "assistant" and turn["kind"] == "update")
    for start, end in zip(starts, starts[1:] + [len(actions)]):
        description = next(descriptions)
        result = annotate(LABEL_PROMPT, {"description": description, "labels": LABELS})
        evidence = {item["label_id"]: item["evidence_spans"] for item in result["assignments"]}
        yield {"description": description, "frame_count": end - start, "label_evidence": evidence}


def build(session, annotate):
    """Join at most five complete views when consecutive views share a label."""
    clip = []
    for view in views(session, annotate):
        if clip and (len(clip) == 5 or
                     clip[-1]["label_evidence"].keys().isdisjoint(view["label_evidence"])):
            yield {"system": session["system"], "views": clip}
            clip = []
        clip.append(view)
    if clip:
        yield {"system": session["system"], "views": clip}
