import torch

class AdaptiveNoiseDP:
    def __init__(self, model, noise_multiplier=1.1, max_grad_norm=1.0, alpha=0.5, beta=0.5):
        """
        Initialize the adaptive differential privacy (DP) with adaptive noise scaling.

        Parameters:
        - model: The trained model
        - noise_multiplier: The amount of noise to add to the gradients
        - max_grad_norm: The maximum gradient norm (gradient clipping)
        - alpha, beta: Hyperparameters controlling noise decay based on gradient magnitude
        """
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.beta = beta

    def apply_dp(self, optimizer, train_loader):
        """
        Apply the adaptive DP to the model during training.

        Parameters:
        - optimizer: The optimizer used for training the model
        - train_loader: A PyTorch DataLoader that provides training data

        Returns:
        - model: The model after training with adaptive DP
        """
        self.model.train()
        for data, target in train_loader:
            optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.loss_function(output, target)

            # Backward pass
            loss.backward()

            # Get the gradients and calculate their variance
            gradients = [param.grad for param in self.model.parameters()]
            gradient_variance = self.calculate_gradient_variance(gradients)

            # Compute adaptive noise based on gradient variance
            noise_scale = self.alpha * gradient_variance + self.beta
            noise = self.noise_multiplier * noise_scale * torch.randn_like(gradients[0])

            # Apply the noise to the gradients
            for param in self.model.parameters():
                param.grad += noise

            # Clip gradients if necessary
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Step the optimizer
            optimizer.step()

        return self.model

    def calculate_gradient_variance(self, gradients):
        """
        Calculate the variance of gradients to control the noise scaling.

        Parameters:
        - gradients: List of gradients from model parameters

        Returns:
        - variance: The variance of the gradients
        """
        flattened_gradients = torch.cat([g.view(-1) for g in gradients])
        return flattened_gradients.var()
