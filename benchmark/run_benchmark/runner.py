"""Free-running inference over public observations in presentation order.

Load one adapter with models.load(name, ...), then call taskN.run(adapter, sample, frames).
Frames are PIL images for chat models, native visual features for streaming models, and
pre-merger features with DeepStack for PathStream. Run with the release root on PYTHONPATH.
"""
def stream(model, task, frames, fps, system, instruction, diagnosis=None):
    records = []
    session = model.start(task, system, instruction)
    try:
        for index, frame in enumerate(frames):
            reply = session.observe(frame, index, fps, model.description_budget)
            kind = "initial_description" if index == 0 else "stream"
            record = {**reply, "kind": kind, "turn_index": index}
            if model.native:
                record["native_decision"] = reply["decision"]
            records.append(record)
            if reply["protocol_error"]:
                return records
        if diagnosis is not None:
            reply = session.diagnose(diagnosis, model.diagnosis_budget)
            records.append({**reply, "kind": "diagnosis", "turn_index": len(records)})
        return records
    finally:
        model.end_case()
