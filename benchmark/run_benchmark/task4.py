"""Task 4: present the ordered clip and answer one public multiple-choice question."""


def run(model, sample, frames):
    prompt = "\n".join([sample["question"], *[f"{k}. {sample['options'][k]}" for k in "ABCD"],
                        sample["instruction"]])
    try:
        result = model.answer_frames(list(frames), prompt, model.choice_budget)
        choice = None if result["protocol_error"] else model.parse_choice(result["response"])
        return {**result, "predicted_answer": choice}
    finally:
        model.end_case()
