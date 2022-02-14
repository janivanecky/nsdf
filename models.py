import torch


class FourierFeatures(torch.nn.Module):
    def __init__(self, embedding_size=256):
        super(FourierFeatures, self).__init__()
        self.fourier_mappings = torch.randn(3, embedding_size // 2)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x):
        x = torch.matmul(2 * torch.pi * x, self.fourier_mappings * self.alpha)
        return torch.cat([torch.sin(x), torch.cos(x)], -1)


class SDFDNN(torch.nn.Module):
    def __init__(
        self,
        size,
        embedding_size,
        hidden_layers=0,
        activation_type="relu",
    ):
        super(SDFDNN, self).__init__()

        if activation_type == "relu":
            activation_layer = torch.nn.ReLU
        elif activation_type == "sigmoid":
            activation_layer = torch.nn.Sigmoid
        else:
            raise ValueError("Invalid activation type.")

        # Input layers.
        layers = [
            FourierFeatures(embedding_size),
            torch.nn.Linear(embedding_size, size),
            activation_layer(),
        ]

        # Hidden layers.
        for _ in range(hidden_layers):
            layers += [torch.nn.Linear(size, size), activation_layer()]

        # Output layer.
        layers += [
            torch.nn.Linear(size, 1),
        ]

        self.layers = torch.nn.Sequential(*layers)

    def __call__(self, x):
        return self.layers(x)
