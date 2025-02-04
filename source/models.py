import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import pytorch_lightning as pl
import timm

from . import warmup_scheduler

class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim=128, lr=0.001, temperature=0.2, weight_decay=0.1, max_epochs=500,
                 warmup_epochs=30, lr_final_value=1e-6,
                 vit="vit_small_patch16_224"):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        self.backbone = timm.create_model(vit,
                                            pretrained=False,
                                            in_chans=5,
                                            num_classes=0)

        # The MLP for g(.) consists of Linear->ReLU->Linear
        embed_dim = self.backbone(torch.rand(1,5,224,224)).shape[-1]
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * hidden_dim),  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = warmup_scheduler.LinearWarmupCosineAnnealingLR(optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            warmup_start_lr=1e-6,
            max_epochs=self.hparams.max_epochs,
            eta_min=self.hparams.lr_final_value
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        # print(f"Shape of imgs: {imgs.shape}")
        # print(f"Type of imgs: {type(imgs)}")
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.mlp(self.backbone(imgs))
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

# model for computing channel means and standard deviations
class ChannelStats(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        batch_mean = torch.mean(x, axis=1)
        batch_sd = torch.std(x, axis=1)
        return batch_mean, batch_sd

class EmbeddingNet(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, x):
        return self.backbone(x)
     
class TransferLearning(nn.Module):
    def __init__(self, model=None):
        super(TransferLearning, self).__init__()
        if model is None:
            model = torchvision.models.resnet50(pretrained=True)
            model =  nn.Sequential(*(list(model.children())[:-1]))
        self.model = model
    
    def forward(self, x):
        # x is an image of B, C, W, H; where B = mini batch size, C is e.g. 5 in the case of 
        # cell painting data
        # we replicate each channel 3 times so it passes throught the network, then concatenate 
        # the resulting embeddings
        emb_list = []
        for c in range(x.shape[1]):
            xrep = x[:, c:c+1, ...].repeat(1, 3, 1, 1)
            emb = self.model(xrep)
            emb = torch.reshape(emb, (emb.shape[0], emb.shape[1]))
            emb_list.append(emb)

        embs_stacked = torch.cat(emb_list, axis=1)
        return embs_stacked
