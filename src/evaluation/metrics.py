def exact_match(predictions, references):
    if not predictions:
        return 0.0
    return sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    ) / len(predictions)
