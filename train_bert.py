import torch
import os
import json
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5' 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, TrainingArguments

def main(resume_from_checkpoint=None):
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print("Loading Bert Model")
    
    
    label2id = {'spam': 0, 'ham': 1}
    id2label = {0: 'spam', 1: 'ham'}
    
    if resume_from_checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(resume_from_checkpoint) 
    else:
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                   num_labels=2,
                                                                   id2label=id2label,
                                                                   label2id=label2id) 
        resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
        )
    print("Loading data...")
    with open('./dataset/train.json', mode='r') as file:
        train_data = json.load(file)
    with open('./dataset/val.json', mode='r') as file:
        val_data = json.load(file)
    train_data = list(train_data)
    val_data = list(val_data)
    train_data+=val_data
    
    for sample in train_data:
        sample["label"] = label2id[sample["label"]]
    # for sample in val_data:
    #     sample["label"] = label2id[sample["label"]]
    
    
    def preprocess_function(examples):
        return tokenizer(examples["input"])

    tokenized_train_set = []
    tokenized_val_set = []
    for sample in train_data:
        tokenized_train_set.append({**preprocess_function(sample),"label": sample["label"]})
    for sample in val_data:
        tokenized_val_set.append({**preprocess_function(sample),"label": sample["label"]})
        
    train_set_size, val_set_size = len(tokenized_train_set), len(tokenized_val_set)
    print(f"Train set has {train_set_size} examples")
    print(f"Val set has {val_set_size} example")
    
    data_collator = DataCollatorWithPadding(
        tokenizer,
    )
    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        output_dir='./out_bert',
        learning_rate=5e-5,
        num_train_epochs=20,
        logging_dir='./runs_bert',
        logging_strategy='steps',
        logging_steps=10,
        # evaluation_strategy='steps',
        # eval_steps=50,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=3,
        save_safetensors=False,
        report_to="tensorboard",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        # eval_dataset=tokenized_train_set,
        train_dataset=tokenized_train_set,
    )
    
    
    if torch.__version__ >= "2":
        model = torch.compile(model)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # trainer.evaluate()
    model.save_pretrained('./out_bert')
    tokenizer.save_pretrained('./out_bert')
    
    
if __name__ == '__main__':
    main()