"""
Standalone inference entrypoint for orbital GNN predictions.

Usage:
    # Local checkpoint:
    python predict.py --checkpoint path/to/checkpoint.pt --input file.log

    # W&B artifact:
    python predict.py --wandb-artifact "entity/project/best-model-fold01:latest" --input file.log

    # Multiple files / directory / JSON output:
    python predict.py --checkpoint ckpt.pt --input f1.log f2.log
    python predict.py --checkpoint ckpt.pt --input-dir path/to/logs/
    python predict.py --checkpoint ckpt.pt --input file.log --output-format json
"""

import argparse
import glob
import json
import os
import sys

import torch

from orbital_gnn import create_orbital_model
from orbital_parser import OrbitalGAMESSParser
from normalization import DataNormalizer


def load_checkpoint(args):
    """Load checkpoint from local path or W&B artifact."""
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"Error: checkpoint file not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading checkpoint from {args.checkpoint}")
        return torch.load(args.checkpoint, map_location='cpu')

    if args.wandb_artifact:
        import wandb
        api = wandb.Api()
        print(f"Downloading W&B artifact: {args.wandb_artifact}")
        artifact = api.artifact(args.wandb_artifact, type='model')
        artifact_dir = artifact.download()
        # Find the .pt file inside the artifact directory
        pt_files = glob.glob(os.path.join(artifact_dir, '*.pt'))
        if not pt_files:
            print(f"Error: no .pt file found in artifact {args.wandb_artifact}", file=sys.stderr)
            sys.exit(1)
        return torch.load(pt_files[0], map_location='cpu')

    print("Error: provide either --checkpoint or --wandb-artifact", file=sys.stderr)
    sys.exit(1)


def build_model_from_checkpoint(checkpoint, device):
    """Reconstruct model from checkpoint config and load weights."""
    config = checkpoint['config']
    model_cfg = config['model']

    # Compute orbital_input_dim from feature flags
    orbital_input_dim = 1  # atomic_num always present
    if model_cfg.get('include_orbital_type', True):
        orbital_input_dim += 1
    if model_cfg.get('include_m_quantum', True):
        orbital_input_dim += 1

    model = create_orbital_model(
        orbital_input_dim=orbital_input_dim,
        hidden_dim=model_cfg.get('hidden_dim', 64),
        num_layers=model_cfg.get('num_layers', 2),
        dropout=model_cfg.get('dropout', 0.25),
        global_pooling_method=model_cfg.get('global_pooling_method', 'sum'),
        orbital_embedding_dim=model_cfg.get('orbital_embedding_dim', 32),
        use_rbf_distance=model_cfg.get('use_rbf_distance', False),
        num_rbf=model_cfg.get('num_rbf', 50),
        rbf_cutoff=model_cfg.get('rbf_cutoff', 5.0),
        include_hybridization=model_cfg.get('include_hybridization', True),
        include_orbital_type=model_cfg.get('include_orbital_type', True),
        include_m_quantum=model_cfg.get('include_m_quantum', True),
        use_element_baselines=model_cfg.get('use_element_baselines', False),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def build_normalizer_from_checkpoint(checkpoint):
    """Reconstruct DataNormalizer from stored stats, or return None."""
    ns = checkpoint.get('normalizer_stats')
    if ns is None:
        return None

    normalizer = DataNormalizer(method=ns['method'])
    normalizer.stats = ns['stats']
    return normalizer


def build_parser_from_checkpoint(checkpoint):
    """Build OrbitalGAMESSParser with matching feature flags."""
    config = checkpoint['config']
    model_cfg = config['model']

    return OrbitalGAMESSParser(
        distance_cutoff=4.0,
        debug=False,
        include_orbital_type=model_cfg.get('include_orbital_type', True),
        include_m_quantum=model_cfg.get('include_m_quantum', True),
        global_target_type=model_cfg.get('global_target_type', 'mcscf_energy'),
    )


def run_inference(model, parser, normalizer, filepath, device):
    """Run inference on a single GAMESS .log file and return predictions."""
    data = parser.parse_and_convert(filepath)

    # Add batch vector (required for single-graph inference)
    data = data.to(device)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        preds = model(data)

    # Denormalize if normalizer is available
    if normalizer:
        preds = normalizer.denormalize_predictions(*preds)

    occupation_pred, keibo_pred, energy_pred, s_pct, p_pct, d_pct, f_pct = preds

    return {
        'file': os.path.basename(filepath),
        'occupation': occupation_pred.cpu().numpy().tolist(),
        'kei_bo': keibo_pred.cpu().numpy().tolist(),
        'energy': energy_pred.cpu().item(),
        'hybridization': {
            's_percent': s_pct.cpu().numpy().tolist(),
            'p_percent': p_pct.cpu().numpy().tolist(),
            'd_percent': d_pct.cpu().numpy().tolist(),
            'f_percent': f_pct.cpu().numpy().tolist(),
        },
    }


def print_results(results, fmt='text'):
    """Print prediction results in text or JSON format."""
    if fmt == 'json':
        print(json.dumps(results, indent=2))
        return

    for res in results:
        print(f"\n{'='*60}")
        print(f"File: {res['file']}")
        print(f"{'='*60}")
        print(f"  Energy:     {res['energy']:.6f}")
        occ = res['occupation']
        print(f"  Occupation: [{', '.join(f'{v:.4f}' for v in occ[:8])}{'...' if len(occ) > 8 else ''}]")
        kb = res['kei_bo']
        print(f"  KEI-BO:     [{', '.join(f'{v:.4f}' for v in kb[:8])}{'...' if len(kb) > 8 else ''}]")
        hyb = res['hybridization']
        s = hyb['s_percent']
        p = hyb['p_percent']
        print(f"  Hybrid s%:  [{', '.join(f'{v:.4f}' for v in s[:8])}{'...' if len(s) > 8 else ''}]")
        print(f"  Hybrid p%:  [{', '.join(f'{v:.4f}' for v in p[:8])}{'...' if len(p) > 8 else ''}]")


def main():
    ap = argparse.ArgumentParser(description='Orbital GNN inference on GAMESS .log files')

    # Checkpoint source (mutually exclusive)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=str, help='Path to local checkpoint .pt file')
    source.add_argument('--wandb-artifact', type=str, help='W&B artifact reference (entity/project/name:version)')

    # Input source
    ap.add_argument('--input', nargs='+', type=str, help='One or more GAMESS .log files')
    ap.add_argument('--input-dir', type=str, help='Directory of GAMESS .log files')

    # Output
    ap.add_argument('--output-format', choices=['text', 'json'], default='text', help='Output format (default: text)')
    ap.add_argument('--device', type=str, default=None, help='Device (cuda/cpu, auto-detected if omitted)')

    args = ap.parse_args()

    # Collect input files
    input_files = []
    if args.input:
        input_files.extend(args.input)
    if args.input_dir:
        input_files.extend(sorted(glob.glob(os.path.join(args.input_dir, '*.log'))))

    if not input_files:
        print("Error: no input files provided. Use --input or --input-dir.", file=sys.stderr)
        sys.exit(1)

    # Device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint and reconstruct components
    checkpoint = load_checkpoint(args)
    model = build_model_from_checkpoint(checkpoint, device)
    normalizer = build_normalizer_from_checkpoint(checkpoint)
    parser = build_parser_from_checkpoint(checkpoint)

    fold_idx = checkpoint.get('fold_index', '?')
    epoch = checkpoint.get('epoch', '?')
    best_mse = checkpoint.get('best_val_kei_bo_mse', '?')
    print(f"Model loaded (fold={fold_idx}, epoch={epoch}, best_val_kei_bo_mse={best_mse})")
    print(f"Running inference on {len(input_files)} file(s)...")

    # Run inference
    results = []
    for filepath in input_files:
        try:
            res = run_inference(model, parser, normalizer, filepath, device)
            results.append(res)
        except Exception as e:
            print(f"Warning: failed on {filepath}: {e}", file=sys.stderr)

    if not results:
        print("Error: all files failed during inference.", file=sys.stderr)
        sys.exit(1)

    print_results(results, args.output_format)


if __name__ == '__main__':
    main()
