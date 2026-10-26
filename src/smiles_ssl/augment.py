"""One corruption family per experiment; sampling never uses labels."""
import torch


def select_mask(shape, fraction, device, eligible=None):
    n, d = shape
    if not 0 < fraction < 1:
        raise ValueError('Corruption fraction must lie in (0,1).')
    scores = torch.rand(n, d, device=device)
    if eligible is not None:
        scores[:, ~eligible] = 2
        available = int(eligible.sum())
    else:
        available = d
    if available == 0:
        raise ValueError('No eligible continuous features for noise.')
    count = max(1, min(available, d - 1, int(round(available * fraction))))
    order = scores.argsort(1)[:, :count]
    return torch.zeros((n, d), dtype=torch.bool, device=device).scatter_(1, order, True)


class Corruptor:
    def __init__(self, modality, config, mean=None, scale=None, noise_columns=None, image_mean=None):
        self.modality, self.config = modality, config
        self.mean, self.scale = mean, scale
        self.noise_columns, self.image_mean = noise_columns, image_mean

    def __call__(self, x, donor=None):
        cfg = self.config
        family, fraction = cfg['family'], cfg['fraction']
        if family not in ('zero', 'mean', 'donor', 'noise'):
            raise ValueError(family)
        if self.modality == 'table':
            eligible = None
            if family == 'noise':
                eligible = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
                eligible[self.noise_columns] = True
            mask = select_mask(x.shape, fraction, x.device, eligible)
            if family == 'zero':
                # Literal zero in source coordinates, then train standardization.
                value = -torch.as_tensor(self.mean, device=x.device, dtype=x.dtype) / torch.as_tensor(self.scale, device=x.device, dtype=x.dtype)
            elif family == 'mean':
                value = torch.zeros_like(x)
            elif family == 'donor':
                if donor is None:
                    raise ValueError('Supply donors drawn only from other training rows.')
                value = donor
            else:
                value = x + torch.randn_like(x) * cfg['sigma']
        else:
            patch = cfg['patch_size']
            h, w = x.shape[-2:]
            if h % patch or w % patch:
                raise ValueError('Image dimensions must be divisible by patch size.')
            mask = select_mask((len(x), (h // patch) * (w // patch)), fraction, x.device)
            mask = mask.reshape(len(x), 1, h // patch, w // patch).repeat_interleave(patch, 2).repeat_interleave(patch, 3)
            if family == 'zero':
                value = torch.zeros_like(x)
            elif family == 'mean':
                value = torch.as_tensor(self.image_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
            elif family == 'donor':
                if donor is None:
                    raise ValueError('Supply donors drawn only from other training images.')
                value = donor  # same spatial location, another train image
            else:
                value = (x + torch.randn_like(x) * cfg['sigma']).clamp(0, 1)
        return torch.where(mask, value, x), mask


def donors(train, indices, modality, device):
    """Uniform donors excluding the current object, independent per table feature."""
    n = len(train)
    if n < 2:
        raise ValueError('At least two train objects required.')
    ids = indices.cpu()
    if modality == 'table':
        d = train.shape[1]
        row = (ids[:, None] + torch.randint(1, n, (len(ids), d))) % n
        return train[row, torch.arange(d)[None, :]].to(device)
    row = (ids + torch.randint(1, n, (len(ids),))) % n
    return torch.stack([train[int(i)] for i in row]).to(device)
