import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

class Quantizer(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel, kernel_size=8, stride=4, padding=2)
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel, kernel_size=4, stride=2, padding=1)
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        blocks.extend(
            [
                nn.ConvTranspose2d(channel, out_channel, kernel_size=8, stride=4, padding=2)
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)





class VQVAE(pl.LightningModule):
    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=2,
            n_res_channel=64,
            embed_dim=64,
            n_embed=512,
            decay=0.99,
            params=None
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantizer(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel
        )
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_b = Quantizer(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel
            # stride=4,
        )

        self.criterion = nn.MSELoss()
        self.latent_loss_weight = 0.25
        if params:
            self.lr = params.lr

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-04)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = y = train_batch[0]
        x_hat, latent_loss = self.forward(x)
        recon_loss = self.criterion(x_hat, y)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss * self.latent_loss_weight
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = y = val_batch[0]
        x_hat, latent_loss = self.forward(x)
        recon_loss = self.criterion(x_hat, y)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss * self.latent_loss_weight
        self.log('val_loss', loss)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        # dec_t = self.dec_t(quant_t)
        # enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_latent(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 2, 3, 1)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 2, 3, 1)

        dec = self.decode(quant_t, quant_b)

        return dec


if __name__ == "__main__":
    rand_input = torch.rand((8, 3, 32, 32))

    model = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512)
    model(rand_input)



