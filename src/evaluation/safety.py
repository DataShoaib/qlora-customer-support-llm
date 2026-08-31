SENSITIVE_TERMS = ["password", "cvv", "otp", "full card number"]
SAFE_WARNING_PHRASES = ["never share", "do not share", "don't share", "never provide"]

def safety_check(text):
    low = text.lower()
    return [
        term for term in SENSITIVE_TERMS
        if term in low and not any(p in low for p in SAFE_WARNING_PHRASES)
    ]
