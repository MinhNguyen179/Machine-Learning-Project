from config import Config
from attack_manager import AttackManager
import random
import random
from datasets import DatasetDict, Dataset
from finetuning.dataset_preparation import prepare_dataset
from finetuning.config import FineTuneConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer



model = AutoModelForCausalLM.from_pretrained('E:\Machine-Learning-Project\outputs\sujet117k-llama2-7b-dp')
tokenizer = AutoTokenizer.from_pretrained(FineTuneConfig.base_model_path)
dataset = prepare_dataset(FineTuneConfig.dataset_path, tokenizer)
train_dataset, val_dataset = prepare_dataset(FineTuneConfig.dataset_path, tokenizer)
train_dataset = dataset['train']
val_dataset  = dataset['val']
train_sample = train_dataset.shuffle(seed=42).select(range(200))
test_sample = val_dataset.shuffle(seed=42).select(range(200))

# Creating the new dataset
combined_data = []

# Add train samples with label 1
for item in train_sample:
    combined_data.append({'input': item['text'], 'label': 1})

# Add test samples with label 0
for item in test_sample:
    combined_data.append({'input': item['text'], 'label': 0})

# Shuffle combined data
random.shuffle(combined_data)

# Convert the list to Dataset
final_dataset = Dataset.from_dict({
    'input': [item['input'] for item in combined_data],
    'label': [item['label'] for item in combined_data]
})

# Create a new DatasetDict for the final dataset
dataset = DatasetDict({
    'train': final_dataset,
    'val': Dataset.from_dict({'input': [], 'label': []}),  # Empty validation dataset as placeholder
    'test': Dataset.from_dict({'input': [], 'label': []})   # Empty test dataset as placeholder
})


# Initialize the manager and execute the attack
attack_manager = AttackManager(
    model=model,
    dataset=dataset,
    attack_type="likelihood",  # Choose from 'likelihood', 'lira', 'neighborhood'
    config=Config
)

metrics = attack_manager.execute_attack()