from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import torch
import random

class NeighborhoodAttack:
    def __init__(self, model, dataset, config, k_neighbors=5):
        """
        Neighborhood-based Membership Inference Attack.
        :param model: The target LLM.
        :param neighbor_model_name: Model used for generating neighbors.
        :param k_neighbors: Number of neighbors to generate for calibration.
        """
        self.model = model
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(config.neighbour_generate_model)
        self.neighbor_model = AutoModelForMaskedLM.from_pretrained(config.neighbour_generate_model)
        self.k_neighbors = k_neighbors

    def generate_neighbors(self, input_text):
        """Generate k semantically-preserving neighbors for the input text."""
        inputs = self.tokenizer(input_text, return_tensors='pt')
        input_ids = inputs['input_ids']
        mask_token_id = self.tokenizer.mask_token_id
        
        neighbors = []
        for _ in range(self.k_neighbors):
            masked_input = input_ids.clone()
            
            num_tokens_to_mask = random.randint(1, len(input_ids[0]) // 3)  
            
            for _ in range(num_tokens_to_mask):
                token_pos = random.randint(1, len(input_ids[0]) - 2)  
                masked_input[0][token_pos] = mask_token_id  
  
            with torch.no_grad():
                outputs = self.neighbor_model(masked_input)
                predictions = outputs.logits


            predicted_ids = torch.argmax(predictions, dim=-1)

            neighbor_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            neighbors.append(neighbor_text)
        
        return neighbors

    def compute_log_likelihood(self, logits, labels):
        """Compute the log-likelihood of a sequence."""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return torch.sum(log_probs.gather(-1, labels.unsqueeze(-1)), dim=-1).mean().item()

    def run(self):
        """
        Run the neighborhood-based attack.
        :param dataset: Dataset to evaluate.
        :return: Scores and labels for evaluation.
        """
        dataset = self.dataset
        scores = []
        labels = []

        for data in dataset:
            inputs, labels_batch = data["input"], data["label"]
            neighbors = self.generate_neighbors(inputs)

            original_logits = self.model(**inputs).logits
            original_log_likelihood = self.compute_log_likelihood(original_logits, labels_batch)

            neighbor_scores = []
            for neighbor in neighbors:
                neighbor_inputs = self.tokenizer(neighbor, return_tensors="pt")
                neighbor_logits = self.model(**neighbor_inputs).logits
                neighbor_scores.append(self.compute_log_likelihood(neighbor_logits, labels_batch))
            average_neighbor_score = np.mean(neighbor_scores)

            neighborhood_score = original_log_likelihood - average_neighbor_score
            scores.append(neighborhood_score)
            labels.append(data["membership_label"])  # 1 for member, 0 for non-member

        return scores, labels