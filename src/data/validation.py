import json
def validate_jsonl(path):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            x=json.loads(line); assert [m.get('role') for m in x['messages']]==['system','user','assistant'], f'Bad schema at {n}'
            rows.append(x)
    return rows
