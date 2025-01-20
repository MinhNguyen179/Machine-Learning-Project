import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class LiRA:
    def __init__(self, model, baseline_model):
        """
        Initialize LiRA with two models: the target model and a baseline model.

        Parameters:
        - model: The target trained model
        - baseline_model: A baseline model, which does not contain the sample in its training data
        """
        self.model = model
        self.baseline_model = baseline_model

    def attack(self, data_loader):
        """
        Perform Likelihood Ratio Attack (LiRA).

        Parameters:
        - data_loader: A PyTorch DataLoader that provides data for inference

        Returns:
        - auc_score: The AUC score for the LiRA
        """
        all_predictions = []
        all_labels = []

        self.model.eval()
        self.baseline_model.eval()

        with torch.no_grad():
            for data, target, membership_label in data_loader:
                model_output = F.softmax(self.model(data), dim=1)
                baseline_output = F.softmax(self.baseline_model(data), dim=1)

                # Compute the likelihood ratio (log-likelihood difference)
                ratio = model_output[:, 1] / (baseline_output[:, 1] + 1e-6)

                all_predictions.extend(ratio.numpy())
                all_labels.extend(membership_label.numpy())

        auc_score = roc_auc_score(all_labels, all_predictions)
        return auc_score
