from datasets import load_dataset
def load_support_dataset(train_path,eval_path):
    return (load_dataset('json',data_files=train_path,split='train'),
            load_dataset('json',data_files=eval_path,split='train'))
