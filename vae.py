import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU(), nn.Linear(400, 200), nn.ReLU()
        )

        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_var = nn.Linear(200, latent_dim)

        # 各ピクセルの値がベルヌーイ分布に従うとモデル化し，モデルは各ピクセルのベルヌーイ分布のパラメタをモデル化する
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)  # 分散が負にならないようにするためにlog_varをモデル化する
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var) ** 0.5
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """


        Args:
            x (torch.Tensor): input data

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: loss, reconst_loss, kl_loss
        """
        x_reconst, mu, log_var = self.forward(x)

        # Ez[log P(x|z)]を近似
        reconst_loss = (
            -(torch.log(x_reconst) * x + torch.log(1 - x_reconst) * (1 - x))
            .sum(axis=1)
            .mean(axis=0)
        )

        # KLダイバージェンスを計算
        kl_loss = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(axis=1).mean(
            axis=0
        )

        return reconst_loss + kl_loss, reconst_loss, kl_loss
