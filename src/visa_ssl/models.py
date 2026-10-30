import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision import transforms as T


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class UniformRotation:
    """Yoo et al. SSL-AD: uniformly sample a quarter-turn, including identity."""
    def __call__(self, image):
        return T.functional.rotate(image, int(np.random.choice([-90, 0, 90, 180])))


class CutPaste:
    """Parameters and sampling from jaeminyoo/SSL-AD, commit 0249fa7.

    Operates on RGB [0,1] tensors before normalization. This is the rectangle
    variant (not scar). The copied patch also receives the author's .1 jitter.
    """
    def __init__(self):
        self.jitter = T.ColorJitter(.1, .1, .1, .1)

    def __call__(self, image):
        height, width = image.shape[-2:]
        area = np.random.uniform(.02, .15) * height * width
        aspect = np.random.choice([np.random.uniform(.3, 1.), np.random.uniform(1., 3.3)])
        h, w = int(math.sqrt(area / aspect)), int(math.sqrt(area * aspect))
        # The official code samples coordinates with an exclusive upper bound.
        i, j = np.random.randint(height - h), np.random.randint(width - w)
        patch = self.jitter(image[:, i:i+h, j:j+w])
        i, j = np.random.randint(height - h), np.random.randint(width - w)
        result = image.clone()
        result[:, i:i+h, j:j+w] = patch
        return result


def transform(size, training=False, augmentation='baseline'):
    if augmentation not in ('baseline', 'rotate', 'cutpaste'):
        raise ValueError(f'Unknown augmentation: {augmentation}')
    # Appendix A specifies crop and jitter values; no additional augmentation.
    ops = ([T.RandomResizedCrop(size, scale=(0.395, 1.0)),
            T.ColorJitter(0.275, 0.275, 0.275, 0.1375)] if training
           else [T.Resize((size, size))])
    if training and augmentation == 'rotate':
        ops.append(UniformRotation())
    ops.append(T.ToTensor())
    if training and augmentation == 'cutpaste':
        ops.append(CutPaste())
    return T.Compose(ops)


def normalize(x):
    return (x - x.new_tensor(MEAN)[None, :, None, None]) / x.new_tensor(STD)[None, :, None, None]


def encoder(seed, architecture='resnet18', image_size=224):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if architecture == 'resnet18':
            model = resnet18(weights=None)
            model.fc = nn.Identity()
            model.num_features = 512
        elif architecture == 'deit_tiny':
            import timm
            model = timm.create_model('deit_tiny_patch16_224', pretrained=False,
                                      img_size=image_size, num_classes=0,
                                      global_pool='avg', fc_norm=False,
                                      drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
            # Use explicit attention, avoiding device-specific fused kernels.
            for block in model.blocks:
                block.attn.fused_attn = False
        else:
            raise ValueError(f'Unknown architecture: {architecture}')
        model.ssl_architecture = architecture
    return model


def spatial(model, x):
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    return model.layer4(model.layer3(model.layer2(model.layer1(x))))


def patch_mask(x, patch, ratio):
    b, _, h, w = x.shape
    if h % patch or w % patch:
        raise ValueError('Image dimensions must divide into patches')
    count = (h // patch) * (w // patch)
    hidden = round(count * ratio)
    if not 0 < hidden < count:
        raise ValueError('Need visible and hidden patches')
    order = torch.rand(b, count, device=x.device).argsort(1)
    mask = torch.zeros(b, count, device=x.device)
    mask.scatter_(1, order[:, :hidden], 1)
    return mask.reshape(b, 1, h // patch, w // patch).repeat_interleave(patch, 2).repeat_interleave(patch, 3)


class SSL(nn.Module):
    def __init__(self, backbone, method, config):
        super().__init__()
        self.encoder, self.method, self.config = backbone, method, config
        if method in ('simclr', 'vicreg'):
            h, d = config['projector_hidden'], config['projector_output']
            self.projector = nn.Sequential(nn.Linear(backbone.num_features, h), nn.BatchNorm1d(h), nn.ReLU(),
                                           nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, d))
        elif method == 'masked_reconstruction':
            if backbone.ssl_architecture == 'deit_tiny':
                if config['patch_size'] != 16:
                    raise ValueError('DeiT-Tiny/16 reconstruction requires patch size 16')
                self.decoder = nn.Sequential(nn.Linear(192, 512), nn.GELU(), nn.Linear(512, 16*16*3))
                return
            layers = []
            channels = [512, 128, 64, 32, 16, 3]
            for a, b in zip(channels[:-1], channels[1:]):
                layers.append(nn.ConvTranspose2d(a, b, 4, 2, 1))
                if b != 3:
                    layers.append(nn.ReLU())
            self.decoder = nn.Sequential(*layers)
        else:
            raise ValueError(method)

    def forward(self, batch):
        c = self.config
        if self.method == 'masked_reconstruction':
            mask = patch_mask(batch, c['patch_size'], c['mask_ratio'])
            hidden_input = normalize(batch) * (1 - mask)
            if self.encoder.ssl_architecture == 'deit_tiny':
                tokens = self.encoder.forward_features(hidden_input)[:, self.encoder.num_prefix_tokens:]
                p = c['patch_size']
                b, _, h, w = batch.shape
                pixels = self.decoder(tokens).reshape(b, h//p, w//p, p, p, 3)
                reconstruction = pixels.permute(0, 5, 1, 3, 2, 4).reshape(b, 3, h, w)
            else:
                reconstruction = self.decoder(spatial(self.encoder, hidden_input))
            loss = ((reconstruction.float() - batch.float()).square() * mask).sum() / (3 * mask.sum())
            return loss, {'masked_mse': loss.detach()}
        x, y = batch
        z1, z2 = self.projector(self.encoder(normalize(x))), self.projector(self.encoder(normalize(y)))
        # Matmul is otherwise re-cast to fp16 even after .float(), which can
        # overflow VICReg covariance and compromise the comparison.
        with torch.autocast(device_type=z1.device.type, enabled=False):
            return self.joint_loss(z1.float(), z2.float())

    def joint_loss(self, z1, z2):
        c = self.config
        if self.method == 'simclr':
            n = len(z1)
            z = F.normalize(torch.cat([z1, z2]), dim=1)
            logits = z @ z.T / c['temperature']
            logits.fill_diagonal_(-float('inf'))
            labels = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
            loss = F.cross_entropy(logits, labels)
            return loss, {'nt_xent': loss.detach()}
        invariance = F.mse_loss(z1, z2)
        std = (F.relu(1 - (z1.var(0) + 1e-4).sqrt()).mean() +
               F.relu(1 - (z2.var(0) + 1e-4).sqrt()).mean()) / 2
        covariance = z1.new_tensor(0.)
        for z in (z1, z2):
            z = z - z.mean(0)
            cov = z.T @ z / (len(z) - 1)
            off = ~torch.eye(cov.shape[0], device=z.device, dtype=torch.bool)
            covariance = covariance + cov[off].square().sum() / cov.shape[0]
        a, b, d = c['vicreg_coefficients']
        loss = a * invariance + b * std + d * covariance
        return loss, {'invariance': invariance.detach(), 'variance': std.detach(), 'covariance': covariance.detach()}
