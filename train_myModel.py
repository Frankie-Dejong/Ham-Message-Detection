import torch
from torch.utils.data import DataLoader
import os
import json
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,4,5' 
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from my_model.model import Model
from my_model.trainer import Trainer

def main():
    print("Loading Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('/media/nas/jiaqi/interaction_style/code/code_myk/train_t5/flan-t5-base')
    print("Loading T5 Model")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained('/media/common/myk/flan-t5-base') 
    t5_model.eval()
    
    print("Loading data...")
    with open('./dataset/train.json', mode='r') as file:
        train_data = json.load(file)
    with open('./dataset/val.json', mode='r') as file:
        val_data = json.load(file)
    train_data = list(train_data)
    val_data = list(val_data)
    
    print(f"Train set has {len(train_data)} samples")
    print(f"Val set has {len(val_data)} samples")
    
    label2id = {'spam': 0, 'ham': 1}
    
    train_list = [(sample['input'], label2id[sample['label']]) for sample in train_data]
    train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
    val_list = [(sample['input'], label2id[sample['label']]) for sample in val_data]
    val_loader = DataLoader(val_list, batch_size=1, shuffle=False)
    
    model = Model()
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        tokenizer,
        t5_model,
        num_epochs=15,
        lr=1e-4
    )
    trainer.train()

    
if __name__ == '__main__':
    main()