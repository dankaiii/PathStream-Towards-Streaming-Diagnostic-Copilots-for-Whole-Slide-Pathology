"""Read the final multiple-choice answer without selecting letters from reasoning."""
import re

MAX_NEW_TOKENS = 4096


def parse_answer(response):
    text = re.split(r"</think>", response, flags=re.I)[-1].strip()
    if re.search(r"<think>", text, flags=re.I):
        return None
    answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if answers:
        text = answers[-1].strip()
    text = text.replace("**", "").replace("__", "").replace("`", "").strip()
    boxed = re.findall(r"\\boxed\{\s*([ABCD])\s*\}", text)
    if boxed:
        return boxed[-1] if len(set(boxed)) == 1 else None
    if answers:
        choice = re.match(r"[\[(]?([ABCD])[\])]?(?=[\s.,:]|$)", text)
        if choice:
            return choice[1]
    # Answer assertions take precedence over options mentioned while comparing alternatives.
    final = re.findall(
        r"\b(?:final\s+|correct\s+)?answer\s*(?:is\s*)?[:=\-]?\s*"
        r"(?:(?:therefore|thus|hence)\s+)?(?:option\s+|choice\s+)?"
        r"[\[(]?([ABCD])(?=[\s\]).,:;]|$)", text, flags=re.I,
    )
    if final:
        return final[-1].upper()
    last_line = text.splitlines()[-1].strip() if text else ""
    choice = re.fullmatch(r"[\[(]?([ABCD])[\])]?\s*[.!]?", last_line)
    if choice:
        return choice[1]
    selected = re.findall(
        r"\b(?:option|choice)\s+([ABCD])\s+"
        r"(?:correctly\b|(?:best\s+)?(?:matches|aligns|corresponds)\b|"
        r"is\s+(?:therefore\s+)?(?:the\s+)?(?:correct|best)\b)", text, flags=re.I,
    )
    if selected and len({letter.upper() for letter in selected}) == 1:
        return selected[-1].upper()
    conclusion = re.findall(
        r"\b(?:therefore|thus|hence)[,:]?\s+(?:option\s+|choice\s+)?"
        r"[\[(]?([ABCD])(?=[\s\]).,:;]|$)", text, flags=re.I,
    )
    return conclusion[-1].upper() if conclusion else None
