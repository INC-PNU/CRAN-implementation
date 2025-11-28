import torch

## NOTE Add GPU Processing wih Torch
####################################
"""
NOTE Bam V3
This is the same structure of BAM and Multi BAM V1/V2
support for batch training
And the most importantly, add torch instead of classical numpy
its increase the speed while maintain the performace
"""
class BAMv3:
    def __init__(self, input_dim, output_dim, eta=1e-5, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.W = torch.empty(output_dim, input_dim, device=self.device)
        torch.nn.init.uniform_(self.W, -0.01, 0.01)

    def _output_function(self, Wx):
        return Wx  

    def train(self, X, num_epochs=1, batch_size=32, verbose=True):
        # n_samples = X.shape[0]
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples = X.shape[0]
        losses = []

        for epoch in range(num_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            X = X[perm]

            for i in range(0, n_samples, batch_size):
                batch = X[i:i+batch_size]
                batch_errors = []

                for x in batch:
                    x = x.view(1, -1)
                    y = self._output_function(self.W @ x.T)
                    x_reconstructed = self._output_function(self.W.T @ y)

                    error = x - x_reconstructed.T
                    batch_errors.append(torch.mean(error ** 2).item())

                    self.W += self.eta * (y @ error)

                    if torch.isnan(self.W).any():
                        raise ValueError("NaN detected in weights!")

                # average error for this batch
                batch_mse = sum(batch_errors) / len(batch_errors)
                losses.append(batch_mse)

                if verbose and i % (batch_size * 10) == 0:
                    print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, MSE = {batch_mse:.6f}")

        return losses

    def compress(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = self._output_function(self.W @ X.T).T
        return y.detach().cpu().numpy()

    def decompress(self, compressed_X):
        Y = torch.tensor(compressed_X, dtype=torch.float32, device=self.device)
        X_reconstructed = self._output_function(self.W.T @ Y.T).T
        return X_reconstructed.detach().cpu().numpy()
    
class MultiBAMv3:
    def __init__(self, layers_dims, eta=1e-4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bams = [
            BAMv3(layers_dims[i], layers_dims[i + 1], eta, self.device)
            for i in range(len(layers_dims) - 1)
        ]

    def train(self, X, num_epochs=1, batch_size=32):
        all_losses = []

        for i, bam in enumerate(self.bams):
            print(f"\n--- Training Layer {i+1}/{len(self.bams)} ---")
            losses = bam.train(X, num_epochs=num_epochs, batch_size=batch_size)
            all_losses.append(losses)
            X = bam.compress(X)  # feed compressed output to next layer

        return all_losses

    def compress(self, X):
        for bam in self.bams:
            X = bam.compress(X)
        return X

    def decompress(self, X):
        for bam in reversed(self.bams):
            X = bam.decompress(X)
        return X 
    
