"""Extract only official images used by the frozen manifest; hash original bytes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request


def prepare(manifest, output, archive=None, url=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    needed = {row['path'] for groups in manifest['categories'].values()
              for split in ('probe_train', 'validation', 'test') for row in groups[split]}
    hashes = {}
    for relative in sorted(needed):
        path = output / relative
        if path.exists():
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    remaining = needed - hashes.keys()
    if remaining:
        stream = open(archive, 'rb') if archive else urllib.request.urlopen(url, timeout=120)
        with stream, tarfile.open(fileobj=stream, mode='r|*') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                pieces = PurePosixPath(member.name).parts
                # Original tar may include a VisA/ prefix. Select by exact suffix,
                # never extract archive-controlled paths or symlinks.
                relative = next(('/'.join(pieces[i:]) for i in range(len(pieces))
                                 if '/'.join(pieces[i:]) in remaining), None)
                if relative is None:
                    continue
                value = tar.extractfile(member).read()
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + '.tmp')
                tmp.write_bytes(value)
                tmp.replace(path)
                hashes[relative] = hashlib.sha256(value).hexdigest()
                remaining.remove(relative)
                if len(hashes) % 100 == 0:
                    print(f'Extracted {len(hashes)}/{len(needed)} images', flush=True)
                if not remaining:
                    break
    if remaining:
        raise ValueError(f'Missing {len(remaining)} official images: {sorted(remaining)[:5]}')
    audit = {'source_url': url, 'manifest_sha256': manifest['manifest_sha256'],
             'image_count': len(hashes), 'image_sha256': dict(sorted(hashes.items()))}
    (output / 'image_hashes.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'Prepared {len(hashes)} original images at {output}', flush=True)
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--archive', type=Path)
    p.add_argument('--expected-hashes', type=Path)
    p.add_argument('--url', default='https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar')
    a = p.parse_args()
    audit = prepare(json.loads(a.manifest.read_text()), a.output, a.archive, a.url)
    expected = a.expected_hashes or a.manifest.parent / 'image_hashes.json'
    if expected.exists():
        assert audit['image_sha256'] == json.loads(expected.read_text())['image_sha256'], 'Original image bytes differ from frozen hashes'
        print('All image hashes match the frozen official-image inventory', flush=True)
