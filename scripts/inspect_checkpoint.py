"""Read-only inspection of a saved checkpoint's internals on fixed held-out windows.

Writes internals.json, manifest.json and PNG panels for the dashboard. Uses the
pilot's recorded validation windows, float32, eval mode and saved BatchNorm
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
from torch.utils.data import DataLoader

from world_model.data import TrajectoryDataset, preprocess_pixels, normalize_actions
from world_model.introspection import (covariance_spectrum, encoder_maps, gaussianity, gradient_norms,
                                       linear_probe, parameter_norms, predictor_internals, rollout_horizon,
                                       sensitivity, state_digest)
from world_model.model import build_model
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


def load(checkpoint, run_manifest, released):
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if released:
        reference = json.loads(Path('runs/diagnostics/reference_full_source/manifest.json').read_text())
        if sha256(checkpoint) != reference['checkpoint_sha256']:
            raise ValueError('Released checkpoint digest mismatch')
        model, stats, step, config = build_model(), reference['normalization']['action'], None, dict(sigreg_knots=17, sigreg_projections=1024, sigreg_weight=0.09)
        model.load_state_dict(saved, strict=True)
    else:
        if saved['fingerprint'] != run_manifest['fingerprint']:
            raise ValueError('Checkpoint belongs to a different run')
        model, stats, step = build_model(saved['model_config']), saved['action_stats'], saved['step']
        model.load_state_dict(saved['model'], strict=True)
        config = run_manifest['config']
    return model.eval(), stats, step, config


def encode_windows(model, dataset, indices, stats, device, batch=32):
    loader = DataLoader(dataset, batch_size=batch, sampler=list(indices), num_workers=0)
    pixels, embeddings, actions, meta = [], [], [], []
    with torch.no_grad():
        for item in loader:
            x = preprocess_pixels(item['pixels'].to(device))
            embeddings.append(model.encode(x).cpu())
            actions.append(normalize_actions(item['action'].to(device), stats).cpu())
            pixels.append(item['pixels'])
            meta.extend(zip(item['episode'].tolist(), item['start'].tolist()))
    return torch.cat(pixels), torch.cat(embeddings), torch.cat(actions), meta


def frame_rows(dataset, meta, t=0):
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
    maps = encoder_maps(model, preprocess_pixels(frames.to(device)))
    heads = maps['cls_attention'].shape[1]
    fig, axes = figure_grid(PANEL_FRAMES, heads + 3)
    for i in range(PANEL_FRAMES):
        axes[i, 0].imshow(frames[i].permute(1, 2, 0).numpy())
        axes[i, 1].imshow(maps['cls_attention'][i].mean(0).cpu().numpy(), cmap='magma')
        for h in range(heads):
            axes[i, 2 + h].imshow(maps['cls_attention'][i, h].cpu().numpy(), cmap='magma')
        axes[i, heads + 2].imshow(maps['patch_pca'][i].cpu().numpy())
    for j, title in enumerate(['input frame', 'CLS attention (mean)'] + [f'head {h}' for h in range(heads)] + ['patch PCA (RGB)']):
        axes[0, j].set_title(title, fontsize=9)
    fig.suptitle(f'{label}: last-layer CLS-to-patch attention on held-out PushT frames ({maps["grid"]}×{maps["grid"]} patches)', fontsize=10)
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
    fig.suptitle(f'{label}: predictor causal attention over the 3-frame history (rows: query, cols: key), batch mean', fontsize=9)
    fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)


def neighbour_panel(model, pixels, z, actions, dataset, meta, device, source, path, label, queries=4):
    """Nearest held-out frames to the predicted next latent, excluding the query window."""
    bank = z.flatten(0, 1)
    window_of = torch.arange(len(z)).repeat_interleave(z.shape[1])
    with torch.no_grad():
        predicted = model.predict(z[:queries, :3].to(device), actions[:queries, :3].to(device))[:, -1].cpu()
    fig, axes = figure_grid(queries, 4)
    rows = frame_rows(dataset, meta)
    for i in range(queries):
        distances = (bank - predicted[i]).square().sum(-1)
        distances[window_of == i] = float('inf')
        nn_pred = int(distances.argmin())
        truth = (bank - z[i, 3]).square().sum(-1)
        truth[window_of == i] = float('inf')
        nn_true = int(truth.argmin())
        panels = [pixels[i, 2], pixels[i, 3], pixels[nn_pred // z.shape[1], nn_pred % z.shape[1]], pixels[nn_true // z.shape[1], nn_true % z.shape[1]]]
        for j, frame in enumerate(panels):
            axes[i, j].imshow(frame.permute(1, 2, 0).numpy())
        axes[i, 0].set_ylabel(f'window {i}', fontsize=8)
    for j, title in enumerate(['current frame (t=2)', 'true next frame (t=3)', 'nearest frame to predicted latent', 'nearest frame to true latent']):
        axes[0, j].set_title(title, fontsize=8)
    fig.suptitle(f'{label}: nearest held-out frames (other windows) in latent space; no decoder is used', fontsize=9)
    fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run', default='runs/diagnostics/pusht_broader_pilot', help='training run whose manifest defines windows')
    parser.add_argument('--output', required=True)
    parser.add_argument('--released', action='store_true')
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = Path(args.run)
    manifest = json.loads((run / 'manifest.json').read_text())
    digest = sha256(args.checkpoint)
    model, stats, step, config = load(args.checkpoint, manifest, args.released)
    model = model.to(device)
    before = state_digest(model)
    label = args.label or ('released weights' if args.released else f'{run.name} step {step}')
    output = Path(args.output); (output / 'panels').mkdir(parents=True, exist_ok=False)
    data_path = manifest['dataset']['path']
    val = TrajectoryDataset(data_path, manifest['val_episodes'], frameskip=5, num_steps=4)
    train = TrajectoryDataset(data_path, manifest['train_episodes'], frameskip=5, num_steps=4)
    long = TrajectoryDataset(data_path, manifest['val_episodes'], frameskip=5, num_steps=HORIZON + 1)
    generator = torch.Generator().manual_seed(0)
    probe_windows = torch.randperm(len(train), generator=generator)[:PROBE_WINDOWS].tolist()
    rollout_windows = torch.randperm(len(long), generator=generator)[:ROLLOUT_WINDOWS].tolist()
    val_windows = manifest['validation_window_indices']

    pixels, z, actions, meta = encode_windows(model, val, val_windows, stats, device)
    flat = z.flatten(0, 1)
    spectrum = covariance_spectrum(flat)
    gauss = gaussianity(flat, seed=0)
    batch_z, batch_a = z[:128, :3].to(device), actions[:128, :3].to(device)
    predictor = predictor_internals(model, batch_z, batch_a)
    sens = sensitivity(model, batch_z, batch_a, directions=8, seed=0)
    encoder = encoder_panel(model, pixels, device, output / 'panels' / 'encoder_attention.png', label)
    spectrum_panel(spectrum, gauss, output / 'panels' / 'spectrum_qq.png', label)
    predictor_panel(predictor, output / 'panels' / 'predictor_attention.png', label)
    with h5py.File(data_path, 'r') as source:
        neighbour_panel(model, pixels, z, actions, val, meta, device, source, output / 'panels' / 'nearest_neighbours.png', label)
        train_pixels, z_train, _, train_meta = encode_windows(model, train, probe_windows, stats, device)
        y_train = state_targets(source['state'][:][frame_rows(train, train_meta)])
        y_val = state_targets(source['state'][:][frame_rows(val, meta)])
    del train_pixels
    probe = linear_probe(z_train[:, 0], y_train, z[:, 0], y_val, targets=list(STATE_TARGETS))
    regularizer = SIGReg(knots=config['sigreg_knots'], num_proj=config['sigreg_projections']).to(device)
    torch.manual_seed(0)
    grads = gradient_norms(model, preprocess_pixels(pixels[:GRADIENT_WINDOWS].to(device)), actions[:GRADIENT_WINDOWS].to(device), regularizer, config["sigreg_weight"])
    params = parameter_norms(model)
    horizon_sums, horizon_count = {}, 0
    loader = DataLoader(long, batch_size=32, sampler=rollout_windows, num_workers=0)
    for item in loader:
        result = rollout_horizon(model, preprocess_pixels(item['pixels'].to(device)), normalize_actions(item['action'].to(device), stats))
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
        dict(id='encoder_attention', title='Encoder attention and patch PCA on held-out frames', caption='Last-layer CLS-to-patch attention per head and RGB PCA of patch tokens over the same frames.', path='panels/encoder_attention.png'),
        dict(id='spectrum_qq', title='Embedding covariance spectrum and Gaussianity', caption='Log eigenvalues of the validation embedding covariance and a pooled Q–Q plot against the SIGReg Gaussian target.', path='panels/spectrum_qq.png'),
        dict(id='predictor_attention', title='Predictor attention over the history', caption='Causal attention per layer and head, averaged over 128 held-out windows.', path='panels/predictor_attention.png'),
        dict(id='nearest_neighbours', title='Nearest held-out frames to predicted latents', caption='Which real frames the one-step prediction lands next to, versus the true next latent; the query window is excluded.', path='panels/nearest_neighbours.png'),
    ]
    training_run = None if args.released else run.resolve().relative_to(Path('runs').resolve()).as_posix()
    write_json(output / 'manifest.json', dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest, step=step, released=args.released, training_run=training_run,
        source_run_manifest=str(run / 'manifest.json'), dataset=manifest['dataset'], precision='float32', mode='eval; saved BatchNorm buffers',
        validation_windows=len(val_windows), embedding_samples=int(len(flat)), predictor_batch=128, gradient_batch=GRADIENT_WINDOWS,
        probe_windows=PROBE_WINDOWS, rollout_windows=ROLLOUT_WINDOWS, rollout_horizon=HORIZON, panel_frames=PANEL_FRAMES,
        seeds=dict(torch=0, sensitivity=0, gaussianity_subsample=0, window_sampling=0), sigreg=dict(knots=config['sigreg_knots'], projections=config['sigreg_projections'], weight=config['sigreg_weight']),
        action_normalization='checkpoint statistics' if not args.released else 'reference full-source scaler',
        protocol='docs/internals-visualization-plan.md', code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        limitation='Descriptive internals on fixed windows; gradients measured in eval mode without an update. Latent-scale quantities are not comparable across separately trained encoders.'))
    write_json(output / 'internals.json', dict(
        step=step, family='released' if args.released else 'pilot', training_run=training_run, label=label,
        checkpoint_sha256=digest, checkpoint_unchanged=True, scalars=scalars,
        series=dict(spectrum=spectrum['eigenvalues'], horizon=horizon, probe_r2=probe['r2'],
                    gate_msa=predictor['gate_msa'].tolist(), gate_mlp=predictor['gate_mlp'].tolist(),
                    predictor_attention_entropy=predictor['attention_entropy'].tolist(),
                    encoder_attention_entropy=encoder['attention_entropy'].tolist(), qq=gauss['qq']),
        panels=panels))
    print(json.dumps(dict(stage='inspected', label=label, step=step, **{k: scalars[k] for k in ('effective_rank', 'shapiro_w_mean', 'gate_msa_mean', 'sensitivity_action_over_state', 'probe_r2_mean', 'horizon_ratio_at_8')}), default=float), flush=True)


if __name__ == '__main__':
    main()
