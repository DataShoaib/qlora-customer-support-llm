def run_qualitative(prompts,generate_fn): return [{'prompt':p,'response':generate_fn(p)} for p in prompts]
