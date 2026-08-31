REQUIRED_ROLES = {"system", "user", "assistant"}

def validate_records(records):
    if not records:
        raise ValueError("Dataset is empty.")
    for i, record in enumerate(records):
        if not isinstance(record.get("messages"), list):
            raise ValueError(f"Record {i}: missing messages list.")
        roles = {m.get("role") for m in record["messages"]}
        if not REQUIRED_ROLES.issubset(roles):
            raise ValueError(f"Record {i}: missing required roles.")
        for message in record["messages"]:
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"Record {i}: empty message content.")
    return True
