from ising_models.ising_model import PairwiseIsingModelInferencer, PairwiseIsingModel
import numpy as np

class TrainingContext():
    def __init__(self, epoch = None, epochs = None, loss = None, model = None, l2_loss = None, kl_loss = None):
        self.epoch = epoch
        self.epochs = epochs
        self.loss = loss
        self.model = model
        self.kl_loss = kl_loss
        self.l2_loss = l2_loss

class PairwiseIsingModelTrainer:
    def __init__(self, ising_model: PairwiseIsingModel, inferencer: PairwiseIsingModelInferencer = None):
        self.ising_model = ising_model
        self.inferencer = PairwiseIsingModelInferencer(self.ising_model) if inferencer is None else inferencer


    def loss(self, observation_dataset, kl_weight=1.0, lambda_weight=0.0):
        """
        Compute pseudo-likelihood loss and gradients for Ising model.

        Args:
            observation_dataset: (N x n_sites) numpy array with spins (+1/-1)
            kl_weight: weight for pseudo-likelihood loss term (default 1.0)
            lambda_weight: weight for L2 regularization on parameters (default 0.0)

        Returns:
            total_loss: scalar loss (pseudo-likelihood + regularization)
            grad_J: gradient wrt J matrix
            grad_H: gradient wrt H vector
            pl_loss: pseudo-likelihood loss (no regularization)
            l2_loss: L2 regularization loss
        """
        N, n_sites = observation_dataset.shape
        J = self.ising_model.J
        H = self.ising_model.H

        pl_loss = 0.0
        grad_J = np.zeros_like(J)
        grad_H = np.zeros_like(H)

        # Compute pseudo-likelihood loss and gradients
        for k in range(N):
            s = observation_dataset[k]
            for i in range(n_sites):
                # Calculate theta_i excluding J[i,i]*s[i]
                theta_i = H[i] + np.dot(J[i, :], s) - J[i, i] * s[i]
                si = s[i]
                tanh_theta_i = np.tanh(theta_i)

                # Accumulate loss
                pl_loss += -(si * theta_i - np.log(2 * np.cosh(theta_i)))

                # Accumulate gradients
                grad_H[i] += -(si - tanh_theta_i)
                for j in range(n_sites):
                    if j != i:
                        grad_J[i, j] += -(si * s[j] - s[j] * tanh_theta_i)

        # Average loss and gradients over dataset
        pl_loss /= N
        grad_J /= N
        grad_H /= N

        # Symmetrize gradient for J since J should be symmetric
        grad_J = (grad_J + grad_J.T) / 2

        # L2 regularization loss and gradient
        l2_loss = np.sum(J**2) + np.sum(H**2)
        grad_J += 2 * lambda_weight * J
        grad_H += 2 * lambda_weight * H

        # Total loss
        total_loss = kl_weight * pl_loss + lambda_weight * l2_loss

        return total_loss, grad_J, grad_H, pl_loss, l2_loss


    def train(self, observation_dataset, epochs=5, learning_rate=0.01, epoch_callback=None, configs = None):
        """Train the Ising model using gradient descent."""
        assert observation_dataset.shape[1] == self.ising_model.n_sites

        if self.inferencer is None:
            self.inferencer = PairwiseIsingModelInferencer(self.ising_model)
        

        for epoch in range(epochs):
            self.inferencer.update_partition_function(configs=configs)

            # Compute the loss and gradients
            loss, grad_J, grad_H, kl_loss, l2_loss = self.loss(observation_dataset)

            # Update the model parameters
            self.ising_model.J -= learning_rate * grad_J
            self.ising_model.H -= learning_rate * grad_H

            # Print the loss for monitoring
            if epoch_callback is not None:
                ctx = TrainingContext(epoch, epochs, loss=loss, model=self.ising_model, l2_loss=l2_loss, kl_loss=kl_loss)
                epoch_callback(ctx)
                
        return self.ising_model