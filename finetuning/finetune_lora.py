import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from dataset_preparation import prepare_dataset

# Load configuration
from config import FineTuneConfig


def fine_tune_lora():
    model = AutoModelForCausalLM.from_pretrained(FineTuneConfig.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(FineTuneConfig.base_model_path)
    dataset = prepare_dataset(FineTuneConfig.dataset_path, tokenizer)
    train_dataset, val_dataset = prepare_dataset(FineTuneConfig.dataset_path, tokenizer)
    train_dataset = dataset['train']
    val_dataset  = dataset['val']

    lora_config = LoraConfig(
        r=FineTuneConfig.lora_r,
        lora_alpha=FineTuneConfig.lora_alpha,
        target_modules=FineTuneConfig.target_modules,
        lora_dropout=FineTuneConfig.lora_dropout,
        bias='none'
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=FineTuneConfig.output_dir,
        per_device_train_batch_size=FineTuneConfig.batch_size,
        num_train_epochs=FineTuneConfig.num_epochs,
        logging_dir=FineTuneConfig.logging_dir,
        save_steps=FineTuneConfig.save_steps,
        evaluation_strategy="steps",
        eval_steps=FineTuneConfig.eval_steps,
        logging_steps=FineTuneConfig.logging_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.save_model(FineTuneConfig.output_dir)

if __name__ == "__main__":
    fine_tune_lora()
