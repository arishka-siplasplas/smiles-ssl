import torch
from torch import nn
from torch.nn import functional as F


def nt_xent(a, b, temperature=.2):
    if a.shape != b.shape or len(a) < 2 or temperature <= 0:
        raise ValueError('NT-Xent needs matching batches, N>=2, temperature>0.')
    z = F.normalize(torch.cat([a, b]), dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    target = (torch.arange(len(z), device=z.device) + len(a)) % len(z)
    loss = F.cross_entropy(logits, target)
    return loss, {'nt_xent': loss}


def vicreg(a, b):
    if len(a) < 2:
        raise ValueError('VICReg requires N>=2.')
    inv = F.mse_loss(a, b)
    var = (F.relu(1 - (a.var(0, unbiased=True) + 1e-4).sqrt()).mean()
           + F.relu(1 - (b.var(0, unbiased=True) + 1e-4).sqrt()).mean()) / 2
    ac, bc = a - a.mean(0), b - b.mean(0)
    ca, cb = ac.T @ ac / (len(a) - 1), bc.T @ bc / (len(b) - 1)
    eye = torch.eye(a.shape[1], dtype=torch.bool, device=a.device)
    cov = (ca.masked_fill(eye, 0).square().sum() + cb.masked_fill(eye, 0).square().sum()) / a.shape[1]
    return 25 * inv + 25 * var + cov, {'invariance': inv, 'variance': var, 'covariance': cov}


def masked_mse(prediction, target, mask):
    mask = mask.expand_as(target)
    if not mask.any():
        raise ValueError('Empty reconstruction target mask.')
    return ((prediction - target).square() * mask).sum() / mask.sum()


class ImageDecoder(nn.Module):
    """Decode the same pooled 512D representation used by the probes."""
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.grid_size = size // 32
        self.fc = nn.Linear(512, 128 * self.grid_size * self.grid_size)
        layers = []
        channels = [128, 128, 64, 32, 16, 3]
        for a, b in zip(channels, channels[1:]):
            layers.extend([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                           nn.Conv2d(a, b, 3, padding=1)])
            if b != 3:
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        x = self.layers(self.fc(z).reshape(-1, 128, self.grid_size, self.grid_size))
        return F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False).sigmoid()


class SSLModel(nn.Module):
    def __init__(self, modality, method, input_dim=None, image_size=128, seed=0):
        super().__init__()
        if method not in ('simclr', 'vicreg', 'masked_reconstruction'):
            raise ValueError(method)
        self.method = method
        # Encoder initialization cannot depend on the chosen SSL head.
        torch.manual_seed(seed)
        if modality == 'table':
            self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(),
                                         nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
            dim = 64
        elif modality == 'image':
            from torchvision.models import resnet18
            self.encoder = resnet18(weights=None)
            self.encoder.fc = nn.Identity()
            dim = 512
        else:
            raise ValueError(modality)
        torch.manual_seed(seed + 100000)
        if method == 'masked_reconstruction':
            self.head = (nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, input_dim))
                         if modality == 'table' else ImageDecoder(image_size))
        else:
            self.head = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 64))

    def objective(self, clean, first, mask, second=None):
        a = self.head(self.encoder(first))
        if self.method == 'masked_reconstruction':
            loss = masked_mse(a, clean, mask)
            return loss, {'masked_mse': loss}
        b = self.head(self.encoder(second))
        return nt_xent(a, b) if self.method == 'simclr' else vicreg(a, b)
