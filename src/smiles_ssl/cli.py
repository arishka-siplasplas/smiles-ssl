import argparse
from copy import deepcopy
from pathlib import Path

from .config import default_config, grid, run_id
from .data import ADB_FILES, CATEGORIES, download_adbench, prepare_mvtec, prepare_table
from .io import write_json


def main():
    parser = argparse.ArgumentParser(description='Long SSL pretraining and anomaly classification')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('download-adbench')
    p.add_argument('--output', default='data/adbench')
    p = sub.add_parser('download-mvtec')
    p.add_argument('--output', default='data/mvtec')
    p.add_argument('--category', choices=CATEGORIES)
    p.add_argument('--workers', type=int, default=2)
    p = sub.add_parser('prepare')
    p.add_argument('--modality', choices=['table', 'image'], required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dataset', help='One table dataset or MVTec category; omitted = all')
    p.add_argument('--seed', type=int, default=20261026)
    p.add_argument('--noise-columns', type=int, nargs='+', help='Restrict noise to these 0-based numeric columns')
    p = sub.add_parser('plan')
    p.add_argument('--scope', choices=['primary', 'augmentations'], default='primary')
    p.add_argument('--stage', choices=['full', 'pilot'], default='full')
    p.add_argument('--modality', choices=['table', 'image'])
    p.add_argument('--dataset')
    p.add_argument('--output', required=True)
    p = sub.add_parser('run')
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--resume', action='store_true')
    p = sub.add_parser('evaluate-test')
    p.add_argument('--run', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p = sub.add_parser('analyze')
    p.add_argument('--runs', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'download-adbench':
        print(download_adbench(args.output))
    elif args.command == 'download-mvtec':
        from .download import download_mvtec
        download_mvtec(args.output, [args.category] if args.category else None, args.workers)
    elif args.command == 'prepare':
        if args.noise_columns is not None and (args.modality != 'table' or not args.dataset):
            raise ValueError('--noise-columns requires --modality table and one --dataset.')
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        datasets = [args.dataset] if args.dataset else (list(ADB_FILES) if args.modality == 'table' else CATEGORIES)
        for name in datasets:
            path = out / f'{name}.json'
            if path.exists():
                raise FileExistsError(f'Immutable prepared manifest already exists: {path}')
            m = (prepare_table(Path(args.data_root) / ADB_FILES[name], name, args.seed, args.noise_columns)
                 if args.modality == 'table' else prepare_mvtec(args.data_root, name, args.seed))
            write_json(path, m)
            print(name, m['summary'])
    elif args.command == 'plan':
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        configs = grid(args.scope, args.modality)
        if args.dataset:
            configs = [c for c in configs if c['dataset'] == args.dataset]
        if args.stage == 'pilot':
            configs = [deepcopy(c) for c in configs if c['seed'] == 4101]
            for c in configs:
                c.update(stage='pilot', seed=3101, epochs=20, short_epoch=10,
                         checkpoints=[0, 1, 2, 5, 10, 20], warmup_epochs=2)
        if not configs:
            raise ValueError('No configs match requested dataset/modality.')
        jobs = []
        for c in configs:
            name = run_id(c)
            write_json(out / (name + '.json'), c)
            jobs.append({'id': name, 'config': name + '.json', 'manifest': c['dataset'] + '.json',
                         'modality': c['modality'], 'status': 'planned'})
        write_json(out / 'plan.json', {'scope': args.scope, 'stage': args.stage, 'n_runs': len(jobs), 'jobs': jobs})
        print(f'{len(jobs)} runs planned; nothing launched.')
    elif args.command == 'run':
        from .run import train
        train(args.config, args.manifest, args.data_root, args.output, args.device, args.resume)
    elif args.command == 'evaluate-test':
        from .run import evaluate_test
        evaluate_test(args.run, args.data_root, args.device)
    elif args.command == 'analyze':
        from .analyze import analyze
        analyze(args.runs, args.output)


if __name__ == '__main__':
    main()
