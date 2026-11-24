"""Task 4. Evaluate (release question, parsed prediction or None) for every question."""
from .metrics import percent_mean


def evaluate(cases):
    correct = [
        prediction is not None and not prediction.get("protocol_error")
        and prediction.get("predicted_answer") == question["answer"]
        for question, prediction in cases
    ]
    return {"accuracy": percent_mean(correct)}
