import torch
from attacker.MiA import MIA
from attacker.LiRA import LiRA
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

def load_and_preprocess_data():
    # Load the dataset from Hugging Face
    dataset = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k")
    # Filter out the sentiment analysis portion
    sentiment_data = dataset['train'].filter(lambda x: x['task'] == 'Sentiment Analysis')
    # Sample 10% of the Sentiment Analysis data
    sentiment_data = sentiment_data.shuffle(seed=42).select(range(int(0.1 * len(sentiment_data))))
    # Extract relevant features and labels
    features = sentiment_data['text']
    labels = sentiment_data['label']
    # Perform train-test split (80% train, 20% test) with equal number of member/non-member
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels,
                                                        random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

class SentimentDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        text = self.features[idx]
        label = self.labels[idx]

        sample = torch.tensor(text, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

def create_dataloader(X_train, X_test, y_train, y_test, batch_size=32):
    train_dataset = SentimentDataset(X_train, y_train)
    test_dataset = SentimentDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    train_loader, test_loader = create_dataloader(X_train, X_test, y_train, y_test)

    model = torch.nn.Sequential(torch.nn.Linear(10, 2))

    mia_attack = MIA(model)

    mia_auc = mia_attack.attack(train_loader)
    print(f"MIA AUC Score: {mia_auc}")

    baseline_model = torch.nn.Sequential(torch.nn.Linear(10, 2))
    lira_attack = LiRA(model, baseline_model)

    lira_auc = lira_attack.attack(train_loader)
    print(f"LiRA AUC Score: {lira_auc}")

if __name__ == '__main__':
    main()


