import torch
import numpy as np

class LikelihoodAttack:
    def __init__(self, model, dataset,config):
        """
        Likelihood-based Membership Inference Attack.
        :param model: The target LLM.
        :param threshold: Decision threshold for membership.
        """
        self.model = model
        self.dataset = dataset
        self.config = config

    def compute_log_likelihood(self, logits, labels):
        """Compute the log-likelihood of a sequence."""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return torch.sum(log_probs.gather(-1, labels.unsqueeze(-1)), dim=-1).mean().item()

    def run(self):
        """
        Run the likelihood-based attack.
        :param dataset: Dataset to evaluate.
        :return: Scores and binary membership predictions.
        """
        dataset = self.dataset
        scores = []
        predictions = []
        labels_set = []
        for data in dataset:
            inputs, labels = data["input"], data["label"]
            logits = self.model(**inputs).logits
            log_likelihood = self.compute_log_likelihood(logits, labels) / len(labels)  # Normalize by length
            scores.append(log_likelihood)
            labels_set.append(labels)

        return scores, labels_set