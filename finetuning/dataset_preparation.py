
import os
import torch
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
def create_balanced_subset(df, fraction=0.1):
    new_df = pd.DataFrame(columns=df.columns)
    for label in df['label'].unique():
        subset = df[df['label'] == label].sample(frac=fraction, random_state=42)
        new_df = pd.concat([new_df, subset])
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return new_df
    # Relabel sentiments to a consistent format
def label_sentiment(value):
    if value == 'positive':
        return "Positive"
    elif value == 'negative':
        return "Negative"
    elif value == 'neutral':
        return "Neutral"
    
    
def prepare_dataset(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name)

    # Filter for sentiment_analysis tasks
    sentiment_analysis_data = dataset.filter(lambda example: example['task_type'] == 'sentiment_analysis')
    sentiment_analysis_data = sentiment_analysis_data['train']

    # Create a new dataset with the desired format
    new_dataset = Dataset.from_dict({
        'text': sentiment_analysis_data['user_prompt'],
        'label': sentiment_analysis_data['answer']
    })

    # Convert to Pandas DataFrame for easier inspection
    df = pd.DataFrame(new_dataset)



    df['label'] = df['label'].apply(label_sentiment)
    df['label'] = df['label'].astype('category')
    df['target'] = df['label'].cat.codes
    df.dropna(inplace=True)

    # Create a balanced subset

    balanced_df = create_balanced_subset(df)
    df = balanced_df

    # Split dataset into train, validation, and test sets
    train_end_point = int(df.shape[0] * 0.6)
    val_end_point = int(df.shape[0] * 0.8)
    df_train = df.iloc[:train_end_point, :]
    df_val = df.iloc[train_end_point:val_end_point, :]
    df_test = df.iloc[val_end_point:, :]

    # Convert from Pandas DataFrame to Hugging Face Dataset
    dataset_train = Dataset.from_pandas(df_train.drop('label', axis=1))
    dataset_val = Dataset.from_pandas(df_val.drop('label', axis=1))
    dataset_test = Dataset.from_pandas(df_test.drop('label', axis=1))
    dataset_train_shuffled = dataset_train.shuffle(seed=42)

    # Combine into DatasetDict
    dataset = DatasetDict({
        'train': dataset_train_shuffled,
        'val': dataset_val,
        'test': dataset_test
    })

    # Compute class weights
    class_weights = (1 / df_train.target.value_counts(normalize=True).sort_index()).tolist()
    
    return dataset