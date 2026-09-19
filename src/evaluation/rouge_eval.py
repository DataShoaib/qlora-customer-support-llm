import evaluate
def compute_rouge(predictions,references):
    return evaluate.load('rouge').compute(predictions=predictions,references=references,use_stemmer=True)
