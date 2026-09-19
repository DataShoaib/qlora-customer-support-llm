def compare_prompts(prompts,base_generate_fn,finetuned_generate_fn):
    return [{'prompt':p,'base':base_generate_fn(p),'finetuned':finetuned_generate_fn(p)} for p in prompts]
