"""Pinned mirror of original MVTec category archives; checksums and safe extraction.

The original mydrive download links returned 404 on 2026-10-26. These hashes
identify the mirror bytes, not an independent certification by MVTec.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tarfile
import urllib.request

from .io import digest, write_json

MIRROR_REVISION = 'dc1c313de9de1be8e41fe6818dc43caac8601efa'
ARCHIVES = {
 'bottle': ('726512129cb3b1f47d4f6cd7c7fc9170db7fc209132249c788fec06029cb5dc3', 155880244),
 'cable': ('1e3d59b80a6a8da5adf32fb0cf311c7b964cde146d6920aef9ca66380999a9cd', 504731120),
 'capsule': ('f6a41cb11f1589d552888fe9c43a1adcbcae15b70073e19093322f836d00a2b6', 404252480),
 'carpet': ('d9dd5064515a20bd75cf24d223c570579141abd72388beef892780b74dbaa85d', 740285760),
 'grid': ('d94a9a651cbfaee22c94d06cf8fa8204a590c872362de0489eb34f1645328de8', 160763852),
 'hazelnut': ('a7f4e7d65a60a235def7236a46fcb854d62e6c6917c16b1ec9cb4ff089d3b0da', 617098680),
 'leather': ('704ffa48b1dc2fc44824aa50442d9febc399dd92a5d6fd8156f8aa0373d15f24', 524889948),
 'metal_nut': ('86e3c8e163ceb19146ad3ddc408e85cc34ae02e0da430e07a73bac6bbb48532a', 165414484),
 'pill': ('ac6ce9ec4ec2a5d67c906d82147bd7431980cf00b78f25d63201a5f17e3a2038', 275267152),
 'screw': ('0a27a365a21e472d19b9fb48cb8e4002b49fabe083bff7f4c16fa57d78154f10', 195344332),
 'tile': ('392228153b96d0db9f179d3f7c2d63e131e907154883b57309fcbf798643173c', 351279732),
 'toothbrush': ('fe72a34b4b0d074f84e844be7ef8c4f64f7f02f20e25f19c9e3b4a231c8bea77', 109357096),
 'transistor': ('146b2166c35a1d0cf37ded091366ac01a50338b4ac704632f1239890eaca4449', 402874300),
 'wood': ('3757297ac2a266072eb4f25a97f805f5b37fd583492b20bd709f310736c1ea47', 498057680),
 'zipper': ('7ab46e0e195da145db30052baea3d3a5d14c5f0e710137c968e0423f02cb942d', 159484784),
}


def extract_category(archive, root, category):
    root = Path(root).resolve()
    with tarfile.open(archive, 'r:xz') as tar:
        for member in tar:
            rel = Path(member.name)
            if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != category:
                raise ValueError(f'Unexpected archive path: {member.name}')
            target = root / rel
            if not target.resolve().is_relative_to(root):
                raise ValueError('Archive path escapes output root.')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            else:
                raise ValueError('Links and special files are not accepted in dataset archives.')


def download_mvtec(root, categories=None, workers=2):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    categories = list(ARCHIVES) if categories is None else categories
    def one(category):
        sha, size = ARCHIVES[category]
        done = root / f'{category}.source.json'
        if done.exists():
            print(f'{category}: already downloaded; prepare verifies image hashes', flush=True)
            return
        url = f'https://huggingface.co/datasets/XavierZXY/MvT-tar/resolve/{MIRROR_REVISION}/{category}.tar.xz'
        archive = root / f'{category}.tar.xz'
        start = archive.stat().st_size if archive.exists() else 0
        if start < size:
            headers = {'Range': f'bytes={start}-'} if start else {}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = start > 0 and response.status == 206
                if start and not append:
                    start = 0
                with archive.open('ab' if append else 'wb') as f:
                    written, last_report = start, start
                    while chunk := response.read(4 * 1024 * 1024):
                        f.write(chunk)
                        written += len(chunk)
                        if written - last_report >= 64 * 1024 * 1024:
                            print(f'{category}: {written / 1e6:.0f}/{size / 1e6:.0f} MB', flush=True)
                            last_report = written
        if archive.stat().st_size != size or digest(archive) != sha:
            raise ValueError(f'Archive checksum/size mismatch: {category}')
        extract_category(archive, root, category)
        write_json(done, {'url': url, 'sha256': sha, 'bytes': size, 'mirror_revision': MIRROR_REVISION,
                          'original_source': 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
                          'license': 'CC BY-NC-SA 4.0', 'source_kind': 'third-party mirror of category archives'})
        archive.unlink()  # Only the verified archive created by this downloader.
        print(f'{category}: verified and extracted', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, categories))
