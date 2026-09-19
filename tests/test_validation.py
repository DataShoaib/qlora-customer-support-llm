from src.data.validation import validate_jsonl
def test_schema(): assert len(validate_jsonl('data/processed/train.jsonl'))>0
