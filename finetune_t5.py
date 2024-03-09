import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5' 
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

def main(resume_from_checkpoint=None):
    print("Loading Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('/media/nas/jiaqi/interaction_style/code/code_myk/train_t5/flan-t5-base')
    print("Loading T5 Model")
    
    if resume_from_checkpoint:
        model = AutoModelForSeq2SeqLM.from_pretrained(resume_from_checkpoint) 
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained('/media/common/myk/flan-t5-base') 
        resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
        )
    print("Loading data...")
    train_data = torch.load(os.path.join('dataset', 'train.pt'))
    val_data = torch.load(os.path.join('dataset', 'val.pt'))
    train_set_size, val_set_size = len(train_data), len(val_data)
    print(f"Train set has {train_set_size} examples")
    print(f"Val set has {val_set_size} example")
    label_pad_token_id = -100        
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=5,
        gradient_accumulation_steps=64 // 5 ,
        output_dir='./out',
        learning_rate=5e-5,
        num_train_epochs=25,
        logging_dir='./runs',
        logging_strategy='steps',
        logging_steps=1,
        # evaluation_strategy='steps',
        # eval_steps=30,
        save_strategy='steps',
        save_steps=150,
        save_total_limit=3,
        save_safetensors=False,
        report_to="tensorboard",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data+val_data,
        # eval_dataset=val_data,
    )
    
    
    if torch.__version__ >= "2":
        model = torch.compile(model)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # trainer.evaluate()
    model.save_pretrained('./out')
    tokenizer.save_pretrained('./out')
    
    
if __name__ == '__main__':
    main()