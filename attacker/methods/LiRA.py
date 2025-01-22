import torch
import numpy as np
class LikelihoodRatioAttack:
    def __init__(self, model, dataset, config):
        """
        Likelihood Ratio Attack (LiRA).
        :param model: The fine-tuned target model.
        :param reference_model: The reference model not trained on private data.
        :param threshold: Decision threshold for membership.
        """
        self.model = model
        self.reference_model = config.reference_model_LIRA
        self.dataset = dataset

    def compute_log_likelihood(self, logits, labels):
        """Compute the log-likelihood of a sequence."""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return torch.sum(log_probs.gather(-1, labels.unsqueeze(-1)), dim=-1).mean().item()

    def run(self):
        """
        Run the likelihood ratio attack.
        :param dataset: Dataset to evaluate.
        :return: Scores and binary membership predictions.
        """
        dataset = self.dataset
        scores = []
        predictions = []
        labels_set = []
        for data in dataset:
            inputs, labels = data["input"], data["label"]
            target_logits = self.model(**inputs).logits
            reference_logits = self.reference_model(**inputs).logits

            # Calculate likelihood ratio
            target_log_likelihood = self.compute_log_likelihood(target_logits, labels)
            reference_log_likelihood = self.compute_log_likelihood(reference_logits, labels)
            likelihood_ratio = (target_log_likelihood - reference_log_likelihood) / len(labels)
            labels_set.append(labels)
            scores.append(likelihood_ratio)

        return scores, labels_set