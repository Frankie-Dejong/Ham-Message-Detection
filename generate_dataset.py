from transformers import T5Tokenizer
import os
import json
import torch
from tqdm import tqdm


def format_prompt(prompt):
    return (
        f"Check whether the SMS is spam: {prompt['input']}:"
    )


def tokenize(sample, tokenizer, padding="max_length"):
    # add prefix to the input for t5
    inputs = format_prompt(sample)

    model_inputs = tokenizer(inputs, padding=padding)

    labels = tokenizer(text_target=sample["label"], padding=padding)

    if padding == "max_length":
        labels["input_ids"] = [(l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
        
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



def generate_t5_tokens(
    phase,
    load_ckpt,
):
    
    print("Loading T5 tokenizer....")

    tokenizer = T5Tokenizer.from_pretrained(load_ckpt)

    print("Loaded Tokenizer, current length is {}".format(len(tokenizer)))

    split_path = os.path.join('./dataset', phase+'.json')
    with open(split_path, 'r') as file:
        data = json.load(file)
    prompt_dataset = list(data)

    print("{} set has {} samples".format(phase, len(prompt_dataset)))
    print("Processing {} set...".format(phase))

    dataset = [
        tokenize(prompt, tokenizer)
        for prompt in tqdm(prompt_dataset)
    ]

    tokens_dir = os.path.join('./dataset')
    os.makedirs(tokens_dir, exist_ok=True)
    tokens_path = os.path.join(tokens_dir, f'{phase}.pt')
    torch.save(dataset, tokens_path)
    print("Saved at {}".format(tokens_path))
    
    
    
if __name__ == '__main__':
    generate_t5_tokens('train', load_ckpt='/media/nas/jiaqi/interaction_style/code/code_myk/train_t5/flan-t5-base')
    generate_t5_tokens('val', load_ckpt='/media/nas/jiaqi/interaction_style/code/code_myk/train_t5/flan-t5-base')