import torch
import numpy as np
from sklearn.metrics import roc_auc_score


class MIA:
    def __init__(self, model, threshold=0.5):
        """
        Initialize the MIA attack with the model.

        Parameters:
        - model: The trained model
        - threshold: Probability threshold to decide if a sample is a member
        """
        self.model = model
        self.threshold = threshold

    def attack(self, data_loader):
        """
        Perform Membership Inference Attack.

        Parameters:
        - data_loader: A PyTorch DataLoader that provides data for inference

        Returns:
        - auc_score: The AUC score for the MIA
        """
        all_predictions = []
        all_labels = []

        self.model.eval()

        with torch.no_grad():
            for data, target, membership_label in data_loader:
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)[:, 1].numpy()  # Assuming binary classification

                # Collect predictions and true membership labels
                all_predictions.extend(probs)
                all_labels.extend(membership_label.numpy())

        auc_score = roc_auc_score(all_labels, all_predictions)
        return auc_score
