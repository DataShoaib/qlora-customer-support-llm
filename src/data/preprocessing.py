import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a professional e-commerce customer support assistant. "
    "Be concise, polite, helpful, and never request passwords, CVV, OTPs, "
    "or full payment-card numbers."
)

def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def to_chat_record(question, answer):
    return {"messages": [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":question.strip()},
        {"role":"assistant","content":answer.strip()},
    ]}
