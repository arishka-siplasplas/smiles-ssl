"""Visual audit of fixed training images; no sample selection for the experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from visa_ssl.models import transform
from visa_ssl.run import seed_all


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/visa'))
    p.add_argument('--manifest', type=Path, default=Path('manifests/visa-v2/fixed.json'))
    p.add_argument('--output', type=Path, default=Path('reports/visa-v2-augmentations.png'))
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    fig, axes = plt.subplots(len(manifest['categories']), 4, figsize=(10, 15), layout='constrained')
    rows = []
    for i, (category, groups) in enumerate(manifest['categories'].items()):
        record = next(r for r in groups['probe_train'] if r['label'] == 1)
        with Image.open(args.data / record['path']) as source:
            image = source.convert('RGB')
        rows.append(dict(category=category, path=record['path'], seed=6101+i))
        for j, augmentation in enumerate(['evaluation', 'baseline', 'rotate', 'cutpaste']):
            seed_all(6101+i)
            operation = transform(224, training=augmentation != 'evaluation',
                                  augmentation='baseline' if augmentation == 'evaluation' else augmentation)
            axes[i, j].imshow(operation(image).permute(1, 2, 0).numpy())
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(augmentation)
            if j == 0:
                axes[i, j].set_ylabel(category)
    fig.suptitle('VisA v2: fixed anomalous training images, same crop/jitter within each row')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    args.output.with_suffix('.json').write_text(json.dumps({
        'dataset': 'VisA by Zou et al. (ECCV 2022)',
        'source': 'https://github.com/amazon-science/spot-diff',
        'manifest_sha256': manifest['manifest_sha256'], 'examples': rows}, indent=2) + '\n')
