import re
from collections.abc import Callable

import pandas as pd


def code_switch(text: str) -> str:
    words = text.split()
    replacements = {
        "اچھا": "good",
        "اچھی": "good",
        "بہترین": "excellent",
        "برا": "bad",
        "محبت": "love",
        "نفرت": "hate",
        "very": "بہت",
        "good": "اچھا",
        "bad": "برا",
    }
    return " ".join(replacements.get(word.lower(), word) for word in words)


def spelling_variation(text: str) -> str:
    text = re.sub(r"([aeiou])\1+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(the|is|was|very)\b", lambda match: match.group(1)[:-1], text, flags=re.IGNORECASE)
    text = text.replace("ہے", "he").replace("ہیں", "hain")
    return text


def build_perturbed(frame: pd.DataFrame, transform: Callable[[str], str], name: str) -> pd.DataFrame:
    result = frame.copy()
    result["text"] = result["text"].map(transform)
    result["perturbation"] = name
    return result
