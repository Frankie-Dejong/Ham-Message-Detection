import torch
import os
from tqdm import tqdm
import json
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,4,5' 
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from generate_dataset import format_prompt
from torch.utils.data import DataLoader
from my_model.model import Model
from my_model.trainer import Trainer

def t5():
    print("Loading Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('./out')
    print("Loading T5 Model")
    
    model = AutoModelForSeq2SeqLM.from_pretrained('./out') 
    
    print("Loading data...")
    with open('./submission/submission.json', 'r') as file:
        data = json.load(file)
    prompt_dataset = list(data)
    
    answers = []
    model.eval()
    for sample in tqdm(prompt_dataset):
        prompt = format_prompt(sample)
        tokens = tokenizer(prompt, return_tensors='pt', padding='do_not_pad')
        
        with torch.no_grad():
            output = model.generate(**tokens, max_new_tokens=5)
            answer = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            if answer not in ['ham', 'spam']:
                raise ValueError(f"UnExpected Answer: {answer}")
            else:
                answers.append(answer+'\n')
    
    with open('./submission/submission_t5.txt', mode='w') as file:
        file.writelines(answers)
    return answers
    
    
def bert():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print("Loading bert Model")
    
    model = AutoModelForSequenceClassification.from_pretrained('./out_bert/checkpoint-400') 
    
    print("Loading data...")
    with open('./submission/submission.json', 'r') as file:
        data = json.load(file)
    prompt_dataset = list(data)
    
    answers = []
    model.eval()
    for sample in tqdm(prompt_dataset):
        inputs = tokenizer(sample["input"], return_tensors='pt')
        
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            answer = model.config.id2label[predicted_class_id]
            if answer not in ['ham', 'spam']:
                raise ValueError(f"UnExpected Answer: {answer}")
            else:
                answers.append(answer+'\n')
    
    with open('./submission/submission_bert.txt', mode='w') as file:
        file.writelines(answers) 
    return answers 
    

def my_model():
    print("Loading Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('/media/nas/jiaqi/interaction_style/code/code_myk/train_t5/flan-t5-base')
    print("Loading T5 Encoder")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained('/media/common/myk/flan-t5-base') 
    t5_model.eval()
    
    print("Loading data...")
    with open('./submission/submission.json', mode='r') as file:
        submit_data = json.load(file)
    submit_data = list(submit_data)
    label2id = {'spam': 0, 'ham': 1}
    id2label = {0: 'spam', 1: 'ham'}
    
    submit_list = [(sample['input'], -1) for sample in submit_data]
    submit_loader = DataLoader(submit_list, batch_size=1, shuffle=False)
    
    
    with open('./dataset/train.json', mode='r') as file:
        train_data = json.load(file)
    train_data = list(train_data)
    train_list = [(sample['input'], label2id[sample['label']]) for sample in train_data]
    train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
    
    model = Model()
    
    trainer = Trainer(
        model,
        train_loader,
        submit_loader,
        tokenizer,
        t5_model,
    )
    print("Loading my custom model...")
    trainer.load_checkpoint('./out_my_model/model_00000015')
    answers = trainer.submit()
    submission = [id2label[ans.item()]+'\n' for ans in answers]
    with open('./submission/submission_my.txt', mode='w') as file:
        file.writelines(submission)  
    return submission
    
    
    
    
    

if __name__ == '__main__':
    label2id = {'spam\n': 0, 'ham\n': 1}
    answers_t5 = t5()
    answers_bert = bert()
    answers_my = my_model()
    final = []
    for i in range(len(answers_t5)):
        answer_t5 = label2id[answers_t5[i]]
        answer_bert = label2id[answers_bert[i]]
        answer_my = label2id[answers_my[i]]
        
        mean = (2 * answer_t5 + answer_bert + answer_my) / 4
        if mean >= 0.5:
            final.append('ham\n')
        else:
            final.append('spam\n')
            
        with open('./submission/submission.txt', mode='w') as file:
            file.writelines(final)  