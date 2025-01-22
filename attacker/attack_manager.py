import json
import random
from evaluate import evaluate_attack
from methods.likelihood import LikelihoodAttack
from methods.LiRA import LikelihoodRatioAttack
from methods.neighbourhood import NeighborhoodAttack

class AttackManager:
    def __init__(self, model, dataset, attack_type, config):
        """
        Args:
            model: The target model.
            dataset: The dataset for testing (members and non-members).
            attack_type: The type of attack ('likelihood', 'lira', 'neighborhood').
            config: Configuration object with parameters.
        """
        self.model = model
        self.dataset = dataset
        self.attack_type = attack_type
        self.config = config
        self.attack_instance = self.get_attack_instance()

    def get_attack_instance(self):
        """Return an attack instance based on the type."""
        if self.attack_type == "likelihood":
            return LikelihoodAttack(self.model, self.dataset, self.config)
        elif self.attack_type == "lira":
            return LikelihoodRatioAttack(self.model, self.dataset, self.config)
        elif self.attack_type == "neighborhood":
            return NeighborhoodAttack(self.model, self.dataset, self.config)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def execute_attack(self):
        """Run the attack and evaluate metrics."""
        scores, labels = self.attack_instance.run()
        metrics = evaluate_attack(scores, labels)
        self.save_results(metrics)
        return metrics

    def save_results(self, metrics):
        with open(self.config.output_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Results saved to {self.config.output_file}")