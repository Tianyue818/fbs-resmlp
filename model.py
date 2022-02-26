from torch import nn
import torch
import torch.nn.functional as F
from utils import winner_take_all

class MlpLayer(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.fc1 = nn.Linear(dim, dim * 4)
    self.fc2 = nn.Linear(dim * 4, dim)
  def forward(self, x):
    x = F.gelu(self.fc1(x))
    x = self.fc2(x)
    return x


class Affine(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.alpha = nn.Parameter(torch.ones(dim))
    self.beta = nn.Parameter(torch.zeros(dim))
  def forward(self, x):
    return self.alpha * x + self.beta


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MlpBlock(nn.Module):
    def __init__(self, dim, num_patch, fbs=True, sparsity_ratio=0.5, layer_scale=1e-4):
        super().__init__()
        self.fbs = fbs
        self.sparsity_ratio = sparsity_ratio
        if fbs:
            self.channel_saliency_predictor = nn.Linear(num_patch, num_patch)
            nn.init.kaiming_normal_(self.channel_saliency_predictor.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.channel_saliency_predictor.bias, 1.)
        self.affine = Affine(dim)
        self.linear_patch = nn.Linear(num_patch, num_patch)
        self.fc = MlpLayer(dim)
        self.post_affine = Affine(dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones((dim)), requires_grad=True)
        self.beta = nn.Parameter(layer_scale * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = self.affine(x)
        x_t = x.transpose(1, 2)
        g = self.channel_saliency_predictor(x_t)
        post_mask = winner_take_all(g, self.sparsity_ratio)
        x_t = self.linear_patch(x_t)
        x_t = x_t.transpose(1, 2)
        post_mask = post_mask.transpose(1, 2)
        x = x + self.gamma * x_t
        x = self.post_affine(x)
        x = x * post_mask
        x = x + self.beta * self.fc(x)
        return x, torch.mean(torch.sum(g, dim=-1))

class FBSResMLP(nn.Module):
  def __init__(self, dim, img_size, depth, in_chan, npatches, n_classes=10):
    super().__init__()
    self.patch_projector = PatchEmbed(patch_size=npatches, in_chans=in_chan,
                                      embed_dim=dim)
    self.blocks = nn.ModuleList([
      MlpBlock(dim, (img_size // npatches) ** 2)
      for _ in range(depth)])
    self.affine = Affine(dim)
    self.linear_classifier = nn.Linear(dim, n_classes)

  def forward(self, x):
    B = x.shape[0]
    x = self.patch_projector(x)
    for blk in self.blocks:
        lasso = 0.
        x, g = blk(x)
        lasso += g
    x = self.affine(x)
    x = x.mean(dim=1).reshape(B, -1)
    x = self.linear_classifier(x)
    return x, lasso
