"""Task 1: select short windows by their update/silence pattern."""
from itertools import groupby


INSTRUCTION = (
    "Observe the pathology views in their presented order. Briefly describe "
    "the morphology of each newly presented view in English. Describe the first frame. "
    "Remain silent while a similar view continues after your description. "
    "A finding may be described again in another view; do not restrict descriptions "
    "to facts absent from the entire history. Do not give a diagnosis."
)


def difficulty(value, easy_max, medium_max):
    if value <= easy_max:
        return "easy"
    return "medium" if value <= medium_max else "hard"


def classify(actions):
    """Classify a window starting with an update. Focus indices are clip-relative."""
    runs = [len(list(group)) for _, group in groupby(actions)]
    switches = len(runs) - 1
    focus = list(range(1, len(actions)))
    if switches == 0:
        pattern, level = "continuous_update", difficulty(runs[0], 3, 6)
    elif switches == 1 and runs[1] <= 4:
        if runs[0] == 1:
            pattern, level = "sustained_silence", difficulty(runs[1], 1, 2)
        else:
            pattern, level = "stop_after_burst", difficulty(runs[0], 3, 5)
            focus = [runs[0]]
    elif switches == 2 and runs[0] <= 3 and runs[1] <= 4 and runs[2] <= 3:
        pattern, level = "restart_after_silence", difficulty(runs[1], 1, 2)
        focus = [runs[0] + runs[1]]
    elif switches >= 3:
        pattern, level = "repeated_switching", difficulty(switches, 3, 4)
    else:
        return None
    return {"pattern": pattern, "difficulty": level, "focus_frame_indices": focus}


def build(session):
    actions = session["supervision"]["frame_targets"]
    for start, action in enumerate(actions):
        if action != "update":
            continue
        for end in range(start + 2, min(start + 10, len(actions)) + 1):
            window = actions[start:end]
            pattern = classify(window)
            if pattern:
                yield {
                    "task": 1, "fps": 1, "system": session["system"], "instruction": INSTRUCTION,
                    "start_frame_index": start, "reference_actions": window,
                    "gate_scoring_start_frame_index": 1, **pattern,
                }
