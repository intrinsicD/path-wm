"""Read-only inspection of a saved checkpoint's internals on fixed validation windows.

Writes internals.json, manifest.json and PNG panels for the dashboard. Uses the
training run's recorded validation windows, float32, eval mode and saved BatchNorm
buffers; the checkpoint hash is verified before and after. Run it through
run.py so the offline dashboard refreshes: see docs/internals-visualization-plan.md.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from world_model.data import TrajectoryDataset, preprocess_pixels, normalize_actions
from world_model.introspection import (covariance_spectrum, encoder_maps, gaussianity, gradient_norms,
                                       linear_probe, parameter_norms, predictor_internals, rollout_horizon,
                                       sensitivity, state_digest)
from world_model.model import build_model
from world_model.protocol import prepare_training_data
from world_model.evaluation import evaluate_prediction
from world_model.objective import SIGReg
from world_model.train import write_json

STATE_TARGETS = ('agent x', 'agent y', 'block x', 'block y', 'block angle sin', 'block angle cos', 'agent vx', 'agent vy')
PANEL_FRAMES, HORIZON, PROBE_WINDOWS, ROLLOUT_WINDOWS, GRADIENT_WINDOWS = 4, 8, 1024, 256, 8


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state_targets(state):
    state = np.asarray(state, dtype=np.float64)
    return np.stack([state[:, 0], state[:, 1], state[:, 2], state[:, 3], np.sin(state[:, 4]), np.cos(state[:, 4]),
                     state[:, 5], state[:, 6]], axis=1)


def read_probe_targets(source, rows, dataset_name):
    """Read physical labels at the requested frame identities, preserving order."""
    if dataset_name == 'tworoom':
        key, names = 'proprio', ['agent x', 'agent y']
    elif dataset_name in ('pusht', 'pusht_cchi'):
        key, names = 'state', list(STATE_TARGETS)
    else:
        raise ValueError(f'No physical probe targets defined for {dataset_name}')
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or rows.min() < 0 or rows.max() >= len(source[key]):
        raise ValueError('Invalid physical probe frame rows')
    unique, inverse = np.unique(rows, return_inverse=True)
    values = source[key][unique][inverse]
    return (values[:, :2] if dataset_name == 'tworoom' else state_targets(values)), names


def load(checkpoint, run_manifest, released, reference_path=None):
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if released:
        reference = json.loads(Path(reference_path or 'runs/diagnostics/reference_full_source/manifest.json').read_text())
        if any(reference['dataset'].get(k)!=run_manifest['dataset'].get(k)
               for k in ('name','repo','revision')):
            raise ValueError('Released reference source differs; supply the matching --reference case manifest')
        detail = reference.get('released_reference', reference)
        if sha256(checkpoint) != detail['checkpoint_sha256']:
            raise ValueError('Released checkpoint digest mismatch')
        stats = detail['action_stats'] if 'released_reference' in reference else reference['normalization']['action']
        model = build_model(detail.get('model_config', {}))
        step, config = None, dict(sigreg_knots=17, sigreg_projections=1024, sigreg_weight=0.09)
        model.load_state_dict(saved, strict=True)
    else:
        if saved['fingerprint'] != run_manifest['fingerprint']:
            raise ValueError('Checkpoint belongs to a different run')
        model, stats, step = build_model(saved['model_config']), saved['action_stats'], saved['step']
        model.load_state_dict(saved['model'], strict=True)
        config = run_manifest['config']
    return model.eval(), stats, step, config


def inspection_datasets(manifest):
    """Restore the saved split before interpreting validation-local indices.

    A random-window validation index addresses a Subset, not its full-source
    base. Reconstruct and verify that mapping with the same protocol as training.
    Long-horizon windows are sampled separately from the validation episode
    population; with a random-window split this is the full source population.
    """
    train, val, tr, va, stats, receipt = prepare_training_data(manifest['config'], manifest['dataset'])
    expected = manifest.get('data_protocol', manifest.get('episode_split'))
    if (tr != manifest['train_episodes'] or va != manifest['val_episodes'] or
            stats != manifest['action_stats'] or receipt != expected or
            len(train) != manifest['train_windows'] or len(val) != manifest['val_windows']):
        raise ValueError('Reconstructed inspection split/statistics differ from saved training manifest')
    indices = manifest['validation_window_indices']
    if not indices or len(set(indices)) != len(indices) or any(not 0 <= i < len(val) for i in indices):
        raise ValueError('Invalid saved validation window indices')
    ds = manifest['dataset']
    long = TrajectoryDataset(ds['path'], va, frameskip=ds['frameskip'], num_steps=HORIZON + 1)
    return train, val, long


def encode_windows(model, dataset, indices, stats, device, batch=32):
    loader = DataLoader(dataset, batch_size=batch, sampler=list(indices), num_workers=0)
    pixels, embeddings, actions, meta = [], [], [], []
    with torch.no_grad():
        for item in loader:
            x = preprocess_pixels(item['pixels'].to(device), model.encoder.config.image_size)
            embeddings.append(model.encode(x).cpu())
            actions.append(normalize_actions(item['action'].to(device), stats).cpu())
            pixels.append(item['pixels'])
            meta.extend(zip(item['episode'].tolist(), item['start'].tolist()))
    return torch.cat(pixels), torch.cat(embeddings), torch.cat(actions), meta


def frame_rows(dataset, meta, t=0):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return [int(dataset.offsets[ep]) + start + t * dataset.frameskip for ep, start in meta]


def figure_grid(rows, cols, size=2.2):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size), dpi=110, squeeze=False)
    for axis in axes.flat:
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    return fig, axes


def encoder_panel(model, pixels, device, path, label):
    frames = pixels[:PANEL_FRAMES, 0]
    maps = encoder_maps(model, preprocess_pixels(frames.to(device), model.encoder.config.image_size))
    heads = maps['cls_attention'].shape[1]
    fig, axes = figure_grid(len(frames), heads + 3)
    for i in range(len(frames)):
        axes[i, 0].imshow(frames[i].permute(1, 2, 0).numpy())
        axes[i, 1].imshow(maps['cls_attention'][i].mean(0).cpu().numpy(), cmap='magma')
        for h in range(heads):
            axes[i, 2 + h].imshow(maps['cls_attention'][i, h].cpu().numpy(), cmap='magma')
        axes[i, heads + 2].imshow(maps['patch_pca'][i].cpu().numpy())
    for j, title in enumerate(['input frame', 'CLS attention (mean)'] + [f'head {h}' for h in range(heads)] + ['patch PCA (RGB)']):
        axes[0, j].set_title(title, fontsize=9)
    fig.suptitle(f'{label}: last-layer CLS-to-patch attention on validation frames ({maps["grid"]}×{maps["grid"]} patches)', fontsize=10)
    fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)
    return maps


def spectrum_panel(spectrum, gauss, path, label):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=110)
    eig = np.asarray(spectrum['eigenvalues'])
    axes[0].semilogy(np.arange(1, len(eig) + 1), np.maximum(eig, 1e-12))
    axes[0].set_xlabel('component'); axes[0].set_ylabel('eigenvalue (log)')
    axes[0].set_title(f'covariance spectrum · effective rank {spectrum["effective_rank"]:.1f} / {spectrum["dimension"]}', fontsize=9)
    axes[1].plot(gauss['qq']['theoretical'], gauss['qq']['sample'], '.', ms=4)
    lim = [min(gauss['qq']['theoretical']), max(gauss['qq']['theoretical'])]
    axes[1].plot(lim, lim, 'k--', lw=1)
    axes[1].set_xlabel('Gaussian quantile'); axes[1].set_ylabel('embedding quantile (pooled, standardized)')
    axes[1].set_title(f'Q–Q · mean Shapiro–Wilk W {gauss["shapiro_w_mean"]:.3f}, {gauss["shapiro_fraction_below_0_95"]:.0%} dims < 0.95', fontsize=9)
    fig.suptitle(label, fontsize=10); fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)


def predictor_panel(internals, path, label):
    attention = internals['attention'].cpu().numpy()  # [layers, heads, T, T]
    layers, heads = attention.shape[:2]
    fig, axes = figure_grid(layers, heads, size=1.3)
    for l in range(layers):
        for h in range(heads):
            axes[l, h].imshow(attention[l, h], vmin=0, vmax=1, cmap='viridis')
            if h == 0:
                axes[l, h].set_ylabel(f'layer {l}', fontsize=8)
    fig.suptitle(f'{label}: predictor causal attention over the {attention.shape[-1]}-frame history (rows: query, cols: key), batch mean', fontsize=9)
    fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)


def neighbour_panel(model, pixels, z, actions, device, path, label, queries=4):
    """Nearest validation frames to the predicted next latent, excluding the query window."""
    if len(z) < 2 or z.shape[1] < 2: raise ValueError('Retrieval needs multiple windows and a future frame')
    queries = min(queries, len(z)); history = z.shape[1]-1
    bank = z.flatten(0, 1)
    window_of = torch.arange(len(z)).repeat_interleave(z.shape[1])
    with torch.no_grad():
        predicted = model.predict(z[:queries, :history].to(device), actions[:queries, :history].to(device))[:, -1].cpu()
    fig, axes = figure_grid(queries, 4)
    for i in range(queries):
        distances = (bank - predicted[i]).square().sum(-1)
        distances[window_of == i] = float('inf')
        nn_pred = int(distances.argmin())
        truth = (bank - z[i, history]).square().sum(-1)
        truth[window_of == i] = float('inf')
        nn_true = int(truth.argmin())
        panels = [pixels[i, history-1], pixels[i, history], pixels[nn_pred // z.shape[1], nn_pred % z.shape[1]], pixels[nn_true // z.shape[1], nn_true % z.shape[1]]]
        for j, frame in enumerate(panels):
            axes[i, j].imshow(frame.permute(1, 2, 0).numpy())
        axes[i, 0].set_ylabel(f'window {i}', fontsize=8)
    for j, title in enumerate([f'current frame (t={history-1})', f'true next frame (t={history})', 'nearest frame to predicted latent', 'nearest frame to true latent']):
        axes[0, j].set_title(title, fontsize=8)
    fig.suptitle(f'{label}: nearest validation frames (other windows) in latent space; no decoder is used', fontsize=9)
    fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run', default='runs/diagnostics/pusht_broader_pilot', help='training run whose manifest defines windows')
    parser.add_argument('--output', required=True)
    parser.add_argument('--released', action='store_true')
    parser.add_argument('--reference', help='Frozen released-reference case manifest; required for TwoRoom')
    parser.add_argument('--label', default=None)
    parser.add_argument('--device', choices=['cpu','cuda'], help='Default: CUDA when available')
    parser.add_argument('--probe-windows', type=int, default=PROBE_WINDOWS)
    parser.add_argument('--rollout-windows', type=int, default=ROLLOUT_WINDOWS)
    args = parser.parse_args()
    if args.probe_windows < 2 or args.rollout_windows < 1: parser.error('Positive inspection budgets required; at least two probe windows')
    torch.set_num_threads(4); torch.manual_seed(0)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    run = Path(args.run)
    manifest = json.loads((run / 'manifest.json').read_text())
    digest = sha256(args.checkpoint)
    model, stats, step, config = load(args.checkpoint, manifest, args.released, args.reference)
    model = model.to(device)
    before = state_digest(model)
    image_size = model.encoder.config.image_size
    label = args.label or ('released weights' if args.released else f'{run.name} step {step}')
    output = Path(args.output); (output / 'panels').mkdir(parents=True, exist_ok=False)
    data_path = manifest['dataset']['path']
    train, val, long = inspection_datasets(manifest)
    generator = torch.Generator().manual_seed(0)
    probe_windows = torch.randperm(len(train), generator=generator)[:args.probe_windows].tolist()
    rollout_windows = torch.randperm(len(long), generator=generator)[:args.rollout_windows].tolist()
    val_windows = manifest['validation_window_indices']

    pixels, z, actions, meta = encode_windows(model, val, val_windows, stats, device)
    prediction_loader = DataLoader(val, batch_size=32, sampler=val_windows, num_workers=0)
    prediction = evaluate_prediction(model, prediction_loader, stats, device,
        image_size=image_size, batches=(len(val_windows) + 31) // 32, precision='float32')
    flat = z.flatten(0, 1)
    spectrum = covariance_spectrum(flat)
    gauss = gaussianity(flat, seed=0)
    history = z.shape[1]-1
    batch_z, batch_a = z[:128, :history].to(device), actions[:128, :history].to(device)
    predictor = predictor_internals(model, batch_z, batch_a)
    sens = sensitivity(model, batch_z, batch_a, directions=8, seed=0)
    encoder = encoder_panel(model, pixels, device, output / 'panels' / 'encoder_attention.png', label)
    spectrum_panel(spectrum, gauss, output / 'panels' / 'spectrum_qq.png', label)
    predictor_panel(predictor, output / 'panels' / 'predictor_attention.png', label)
    with h5py.File(data_path, 'r') as source:
        neighbour_panel(model, pixels, z, actions, device, output / 'panels' / 'nearest_neighbours.png', label)
        train_pixels, z_train, _, train_meta = encode_windows(model, train, probe_windows, stats, device)
        y_train, target_names = read_probe_targets(source, frame_rows(train, train_meta), manifest['dataset']['name'])
        y_val, _ = read_probe_targets(source, frame_rows(val, meta), manifest['dataset']['name'])
    del train_pixels
    probe = linear_probe(z_train[:, 0], y_train, z[:, 0], y_val, targets=target_names)
    regularizer = SIGReg(knots=config['sigreg_knots'], num_proj=config['sigreg_projections']).to(device)
    torch.manual_seed(0)
    grads = gradient_norms(model, preprocess_pixels(pixels[:GRADIENT_WINDOWS].to(device), image_size), actions[:GRADIENT_WINDOWS].to(device), regularizer, config["sigreg_weight"])
    params = parameter_norms(model)
    horizon_sums, horizon_count = {}, 0
    loader = DataLoader(long, batch_size=32, sampler=rollout_windows, num_workers=0)
    for item in loader:
        result = rollout_horizon(model, preprocess_pixels(item['pixels'].to(device), model.encoder.config.image_size), normalize_actions(item['action'].to(device), stats))
        n = len(item['pixels']); horizon_count += n
        for key in ('prediction_mse', 'one_step_mse', 'copy_mse', 'previous_mse'):
            horizon_sums[key] = [a + b * n for a, b in zip(horizon_sums.get(key, [0.0] * HORIZON), result[key])]
    horizon = {'horizon': list(range(1, HORIZON + 1)), **{k: [v / horizon_count for v in values] for k, values in horizon_sums.items()}}
    if state_digest(model) != before or sha256(args.checkpoint) != digest:
        raise RuntimeError('Inspection changed the model or checkpoint')
    scalars = {
        **{k: spectrum[k] for k in ('effective_rank', 'rankme', 'participation_ratio', 'top_eigenvalue_fraction', 'per_dim_std_min', 'per_dim_std_median', 'per_dim_std_max')},
        **{k: gauss[k] for k in ('shapiro_w_mean', 'shapiro_w_min', 'shapiro_fraction_below_0_95', 'skewness_mean_abs', 'excess_kurtosis_mean')},
        'embedding_std': float(flat.std(0).mean()),
        'gate_msa_mean': float(predictor['gate_msa'].mean()), 'gate_mlp_mean': float(predictor['gate_mlp'].mean()),
        'action_embedding_norm': predictor['action_embedding_norm'], 'state_norm': predictor['state_norm'],
        'predictor_attention_entropy_mean': float(predictor['attention_entropy'].mean()),
        'encoder_attention_entropy_last': float(encoder['attention_entropy'][-1]),
        'encoder_attention_entropy_mean': float(encoder['attention_entropy'].mean()),
        'sensitivity_action': sens['action'], 'sensitivity_state': sens['state'], 'sensitivity_action_over_state': sens['action_over_state'],
        **{f'param_norm_{k}': v for k, v in params.items()},
        **{f'grad_norm_{k}': v for k, v in grads.items() if k not in ('loss', 'pred_loss', 'sigreg_loss')},
        **{f'grad_batch_{k}': grads[k] for k in ('loss', 'pred_loss', 'sigreg_loss')},
        'probe_r2_mean': probe['r2_mean'],
        'horizon_ratio_at_8': horizon['prediction_mse'][-1] / horizon['copy_mse'][-1] if horizon['copy_mse'][-1] > 0 else None,
    }
    panels = [
        dict(id='encoder_attention', title='Encoder attention and patch PCA on validation frames', caption='Last-layer CLS-to-patch attention per head and RGB PCA of patch tokens over the same frames.', path='panels/encoder_attention.png'),
        dict(id='spectrum_qq', title='Embedding covariance spectrum and Gaussianity', caption='Log eigenvalues of the validation embedding covariance and a pooled Q–Q plot against the SIGReg Gaussian target.', path='panels/spectrum_qq.png'),
        dict(id='predictor_attention', title='Predictor attention over the history', caption=f'Causal attention per layer and head over {history} frames, averaged over {len(batch_z)} validation windows.', path='panels/predictor_attention.png'),
        dict(id='nearest_neighbours', title='Nearest validation frames to predicted latents', caption='Which real frames the one-step prediction lands next to, versus the true next latent; the query window is excluded.', path='panels/nearest_neighbours.png'),
    ]
    training_run = None if args.released else run.resolve().relative_to(Path('runs').resolve()).as_posix()
    population = manifest.get('population', 'Validation episodes disjoint from training episodes')
    index_digest = lambda indices: hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()
    write_json(output / 'manifest.json', dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest, step=step, released=args.released, training_run=training_run,
        source_run_manifest=str(run / 'manifest.json'), dataset=manifest['dataset'], device=str(device), precision='float32', mode='eval; saved BatchNorm buffers',
        validation_windows=len(val_windows), validation_window_indices=val_windows,
        data_protocol=manifest.get('data_protocol', manifest.get('episode_split')), population=population,
        probe_window_indices_sha256=index_digest(probe_windows),
        rollout_window_indices_sha256=index_digest(rollout_windows),
        rollout_population='Source episodes shared with training' if isinstance(val, Subset) else 'Validation episodes disjoint from training',
        embedding_samples=int(len(flat)), predictor_batch=len(batch_z), gradient_batch=min(GRADIENT_WINDOWS,len(pixels)),
        probe_windows=len(probe_windows), rollout_windows=len(rollout_windows), rollout_horizon=HORIZON, panel_frames=min(PANEL_FRAMES,len(pixels)),
        probe_targets=target_names, history=history, image_size=image_size, released_reference_source=args.reference,
        seeds=dict(torch=0, sensitivity=0, gaussianity_subsample=0, window_sampling=0), sigreg=dict(knots=config['sigreg_knots'], projections=config['sigreg_projections'], weight=config['sigreg_weight']),
        action_normalization='checkpoint statistics' if not args.released else 'reference full-source scaler',
        protocol='docs/internals-visualization-plan.md', code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        limitation='Descriptive internals on fixed windows; gradients measured in eval mode without an update. Latent-scale quantities are not comparable across separately trained encoders.'))
    write_json(output / 'prediction.json', prediction)
    write_json(output / 'internals.json', dict(
        step=step, family='released' if args.released else 'local', training_run=training_run, label=label,
        checkpoint_sha256=digest, checkpoint_unchanged=True, scalars=scalars,
        series=dict(spectrum=spectrum['eigenvalues'], horizon=horizon, probe_r2=probe['r2'],
                    gate_msa=predictor['gate_msa'].tolist(), gate_mlp=predictor['gate_mlp'].tolist(),
                    predictor_attention_entropy=predictor['attention_entropy'].tolist(),
                    encoder_attention_entropy=encoder['attention_entropy'].tolist(), qq=gauss['qq']),
        panels=panels))
    print(json.dumps(dict(stage='inspected', label=label, step=step, **{k: scalars[k] for k in ('effective_rank', 'shapiro_w_mean', 'gate_msa_mean', 'sensitivity_action_over_state', 'probe_r2_mean', 'horizon_ratio_at_8')}), default=float), flush=True)


if __name__ == '__main__':
    main()
