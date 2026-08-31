from src.data.validation import validate_records

def test_valid_record():
    record={"messages":[
        {"role":"system","content":"You are helpful."},
        {"role":"user","content":"Hello"},
        {"role":"assistant","content":"Hi"},
    ]}
    assert validate_records([record]) is True
