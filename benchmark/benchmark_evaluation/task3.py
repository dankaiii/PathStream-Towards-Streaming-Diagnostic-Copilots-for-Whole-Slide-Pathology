"""Task 3. Each cohort contains (session, decoded turn records, prepared reference) triples.

Create each fixed diagnosis reference with diagnosis.prepare_reference(session, judge).
The runner marks the final diagnosis prediction with kind="diagnosis".
"""
from .diagnosis import claim_scores
from .metrics import invalid_output, percent_mean, rouge_l_f1, stream_scores


def descriptions(session):
    targets = session["supervision"]["frame_targets"]
    references = [""] * len(targets)
    frame = -1
    for turn in session["conversation"]:
        if turn["role"] == "stream":
            frame += turn["num_frames"]
        elif turn["role"] == "assistant" and turn["kind"] == "update":
            references[frame] = turn["content"]
    return references


def evaluate(trajectory, video, encode, judge):
    result, streams = {}, []
    for name, cases in (("trajectory", trajectory), ("video", video)):
        precision, relevance, rouge = [], [], []
        for session, predictions, reference in cases:
            targets = session["supervision"]["frame_targets"]
            frame_turns = {}
            final = {}
            for turn in predictions:
                if turn.get("kind") == "diagnosis":
                    final = turn
                else:
                    frame_turns[turn["turn_index"]] = turn
            if name == "trajectory":
                streams.append((targets, descriptions(session), frame_turns))

            response = "" if invalid_output(final) else final.get("response", "").strip()
            p, r = claim_scores(reference, response, judge)
            precision.append(p)
            relevance.append(r)
            rouge.append(rouge_l_f1(reference["reference"], response))
        result[name + "_diagnosis"] = {
            "wsi_precision": percent_mean(precision), "wsi_relevance": percent_mean(relevance),
            "rouge_l_f1": percent_mean(rouge),
        }
    return {**stream_scores(streams, encode), **result}
