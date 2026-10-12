"""
გაწერილ წესებზე დაფუძნებული SMS სპამ ფილტერი, ყველა წესს აქვს საკუთარი სახელი და წონა.
SMS-ები ფასდება წონების ჯამით, ანუ ქულით და თუ ის კვეთს დაწესებულ ზღვარს ჩაითვლება სპამად.
წესები დაწერილია და დატუნინგებულია მხოლოდ data/train.csv ის გამოყენებით.

"""

import re

import numpy as np
import pandas as pd

THRESHOLD = 2.0

def _shouting(text: str) -> bool:
    # თუ წინადადებაში ასოების 30% ზე მეტი uppercase არის ესეიგი მესიჯი "ყვირის",
    # წინადადებები რომლებიც შეიცავენ 20 ზე ნაკლებ ქერექთერს არ ექვემდებარევბა ამ წესს
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return False
    upper = sum(c.isupper() for c in letters)
    return upper / len(letters) > 0.30


def _long_message(text: str) -> bool:
    # სპამ მესიჯების სიგრძე საშუალოდ 2 ჯერ მეტია ჰემ მესიჯებზე
    return len(text) > 120


def _many_exclamations(text: str) -> bool:
    return text.count("!") >= 2


def _rx(pattern: str, flags: int = re.IGNORECASE):
    compiled = re.compile(pattern, flags)
    return lambda text: compiled.search(text) is not None



RULES = [
    # ძლიერი მაჩვენებელი იმისა რომ მესიჯი სპამია 2.0
    ("phone_number",   _rx(r"\b0\d{9,10}\b|\b\d{11}\b"), 2.0),
    ("pricing_small_print",
                       _rx(r"\b\d{2,3}p\b|\bper\s+(?:msg|min|wk|week)\b|\bT\s?&\s?Cs?\b|\b18\s?\+"), 2.0),
    ("text_keyword",   _rx(r"\b(?:[Tt]xt|[Tt]ext|[Ss]end|[Rr]eply)\s+[A-Z]{2,}\b", 0), 2.0),
    ("opt_out",        _rx(r"\bopt[\s-]?out\b|\bunsubscribe\b|\b(?:send|txt|text|reply)\s+stop\b"), 2.0),

    # საშუალო მაჩვენებელი 1.0-1.5
    ("money",        _rx(r"£\s?\d"), 1.5),
    ("cash_voucher", _rx(r"\bcash\b|\bvoucher"), 1.0),   
    ("url",            _rx(r"https?://|www\.|\.(?:com|net|co\.uk)\b"), 1.0),
    ("short_code",     _rx(r"\b\d{5}\b"), 1.0),
    ("prize_words", _rx(r"\b(?:prize|winner|won(?!')|win|award(?:ed)?|guaranteed|bonus|reward)\b"), 1.0),
    ("claim_words",    _rx(r"\b(?:claim|redeem|collect)\b"), 1.0),
    ("free",           _rx(r"\bfree\b"), 1.0),
    ("urgency",        _rx(r"\b(?:urgent|immediately|expires?|limited|last\s+chance|final\s+(?:attempt|notice))\b"), 1.0),
    ("mobile_offer",   _rx(r"\b(?:ringtones?|camera\s+phone|video\s+phone|subscription|mobile\s+update)\b"), 1.0),

    # სუსტი მაჩვენებელი 0.5
    ("shouting",       _shouting, 0.5),
    ("long_message",   _long_message, 0.5),
    ("exclamations",   _many_exclamations, 0.5),
]



# აბრუნებს ქულას და წესებს რომლებიც დაირღვა
def score_message(text: str):
    fired = [name for name, check, _ in RULES if check(text)]
    weights = {name: w for name, _, w in RULES}
    score = sum(weights[name] for name in fired)
    return score, fired

# აბრუნებს ფრედიქშენს მესიჯისთვის
def predict(texts, threshold: float = THRESHOLD) -> np.ndarray:
    return np.array([int(score_message(t)[0] >= threshold) for t in texts])

# აბრუნებს მხოლოდ ქულას
def scores(texts) -> np.ndarray:
    return np.array([score_message(t)[0] for t in texts])


def explain(text: str, threshold: float = THRESHOLD) -> str:
    score, fired = score_message(text)
    label = "SPAM" if score >= threshold else "HAM"
    reasons = ", ".join(fired) if fired else "no rules fired"
    return f"{label} (score {score:g} / threshold {threshold:g}): {reasons}"


# white box კრიტერიუმი, ბეჭდავს გადაწყვეტილების ზუსტ მიზეზს 
def rule_stats(texts, y) -> pd.DataFrame:
    """
    დასატუნინგებელი ინსტრუმენტი თუ რამდენჯერ ამოქმედდა წესი სპამზე და ჰემზე.
    დაბალი სიზუტე ნიშნავს რომ წესი ზედმეტად ზოგადია და მისი წონის შემცირება მომიწია
    """
    texts = list(texts)
    y = np.asarray(y)
    rows = []
    for name, check, weight in RULES:
        hits = np.array([check(t) for t in texts])
        on_spam = int((hits & (y == 1)).sum())
        on_ham = int((hits & (y == 0)).sum())
        total = on_spam + on_ham
        rows.append({
            "rule": name,
            "weight": weight,
            "fires_spam": on_spam,
            "fires_ham": on_ham,
            "precision": on_spam / total if total else float("nan"),
            "spam_coverage": on_spam / (y == 1).sum(),
        })
    return pd.DataFrame(rows).sort_values("precision", ascending=False)