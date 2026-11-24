"""Task 3: export complete streaming sessions that end with a reference diagnosis."""


def build(sessions):
    for session in sessions:
        last = session["conversation"][-1]
        if last["role"] == "assistant" and last["kind"] == "diagnosis" and last["content"].strip():
            yield session
