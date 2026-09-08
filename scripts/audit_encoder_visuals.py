"""Verify saved first-seed PCA/attention panels without training or overwriting them."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from world_model.curriculum.data import file_hash, task_frames
from world_model.curriculum.encoder_factorial import ARMS
from world_model.curriculum.encoder_variants import matched_models
from world_model.curriculum.encoder_visual_audit import attention_details, entropy, fit_pca, pca_colors, variance_parts
from world_model.curriculum.pose_accessibility import analyze, setup
from world_model.pusht.checkpoints import json_atomic, read_checkpoint

ROOT = Path('runs/encoder_study_2026-09-08')
OUT = ROOT / 'evaluation/visual_audit'
SOURCE = ROOT


def plot_pca(raw, colors):
    labels = ['Input', 'Original fine: joint PCA', 'Original coarse: joint PCA',
              'Coarse: separate PCA', 'Coarse: image-centered PCA']
    rows = [raw['rgb'].transpose(0, 2, 3, 1), colors['joint_fine'].reshape(-1, 16, 16, 3),
            colors['joint_coarse'].reshape(-1, 8, 8, 3), colors['coarse'].reshape(-1, 8, 8, 3),
            colors['coarse_centered'].reshape(-1, 8, 8, 3)]
    fig, axes = plt.subplots(5, 3, figsize=(7.5, 9), layout='constrained')
    for i, row in enumerate(rows):
        for j in range(3):
            axes[i, j].imshow(row[j], interpolation='nearest')
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f'Test row {raw["test_indices"][j]}', fontsize=10)
        axes[i, 0].set_ylabel(labels[i], fontsize=9)
    fig.suptitle('Deeper/on · seed 7107 · PCA inspection alternatives\nTraining-fixed fits and color bounds; independent rows do not share color meanings', fontsize=10)
    fig.savefig(OUT / 'deeper_pca_comparison.png', dpi=110)
    plt.close(fig)


def plot_attention(raw, attention):
    fig, axes = plt.subplots(2, 4, figsize=(9, 5.2), layout='constrained')
    for row, (direction, query_size, key_size) in enumerate([('fine', 16, 8), ('coarse', 8, 16)]):
        col, y = np.clip((raw['targets'][0, :2] * query_size).astype(int), 0, query_size - 1)
        q = y * query_size + col
        for head in range(4):
            probabilities = attention[direction]['probabilities'][0, head, q].reshape(key_size, key_size)
            # Ratios share the uniform=1 reference despite differing numbers of keys.
            ratio = probabilities * key_size ** 2
            im = axes[row, head].imshow(ratio, cmap='cividis', vmin=0, vmax=4, interpolation='nearest')
            axes[row, head].set_title(f'Head {head + 1} · H={attention[direction]["entropy"][0, head, q]:.4f}', fontsize=10)
            axes[row, head].set_xticks([]); axes[row, head].set_yticks([])
        axes[row, 0].set_ylabel(f'{direction} queries → {"coarse" if row == 0 else "fine"} keys', fontsize=9)
    fig.colorbar(im, ax=axes, shrink=.7, label='Attention probability / uniform probability (clipped at 4)')
    fig.suptitle(f'Deeper/on · seed 7107 · test row {raw["test_indices"][0]}\nQuery at the pusher cell in each grid; head entropy computed before averaging', fontsize=11)
    fig.savefig(OUT / 'deeper_attention_heads.png', dpi=120)
    plt.close(fig)


@torch.no_grad()
def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        raise FileExistsError('preserve completed audit')
    train = task_frames('data/pusht_world_model/cchi_v1', 'train')
    test = task_frames('data/pusht_world_model/cchi_v1', 'test')
    summaries = {}
    for arm, (depth, exchange) in ARMS.items():
        raw_path = SOURCE / 'evaluation/internals' / f'{arm}_states.npz'
        manifest_path = raw_path.with_name(f'{arm}_manifest.json')
        raw = dict(np.load(raw_path))
        manifest = json.loads(manifest_path.read_text())
        checkpoint = Path(manifest['checkpoint'])
        before = file_hash(checkpoint)
        if before != manifest['checkpoint_sha256'] or train.fingerprint != manifest['dataset_fingerprint']:
            raise ValueError('source identity changed')
        models = matched_models(7107, depth, exchange)
        state = read_checkpoint(checkpoint)
        for name, model in models.items():
            model.load_state_dict(state['models'][name]); model.eval().requires_grad_(False)
        bank = []
        for start in range(0, len(raw['train_pca_indices']), 32):
            rgb, _ = train.batch(raw['train_pca_indices'][start:start + 32])
            bank.append(models['E'](rgb).tokens().numpy())
        bank = np.concatenate(bank)
        colors, pca_metrics, pca_arrays = {}, {}, {}
        joint = dict(mean=raw['pca_mean'], basis=raw['pca_basis'], low=raw['color_low'],
                     high=raw['color_high'], image_centered=False)
        joint_training = (bank.reshape(-1, 64).astype('float64') - joint['mean']) @ joint['basis']
        centered_bank = bank.reshape(-1, 64).astype('float64') - joint['mean']
        joint_fraction = float(np.square(joint_training).sum() / np.square(centered_bank).sum())
        np.testing.assert_allclose(joint_fraction, manifest['pca_top3_variance_fraction'], rtol=1e-6)
        for name, part in [('fine', slice(None, 256)), ('coarse', slice(256, None))]:
            training = bank[:, part]
            colors['joint_' + name], clipped = pca_colors(raw[name], joint)
            pca_metrics[name] = dict(training_variance=variance_parts(training),
                                     six_frame_variance=variance_parts(raw[name]), joint_color_clipped_fraction=clipped)
            for centered in (False, True):
                fit = fit_pca(training, centered)
                tag = name + ('_centered' if centered else '')
                colors[tag], clip = pca_colors(raw[name], fit)
                pca_metrics[tag + '_fit'] = dict(top3_variance_fraction=fit['top3_variance_fraction'],
                                                test_color_clipped_fraction=clip)
                for key, value in fit.items():
                    pca_arrays[tag + '_' + key] = value
        attention, incoming, handles = {}, {}, []

        def capture_incoming(name):
            def hook(module, args):
                incoming[name] = args[0].detach().numpy().copy()
            return hook

        def capture_attention(name):
            def hook(module, args, output):
                details = attention_details(module, *args)
                # SDPA and explicit matmul accumulate differently in float32.
                # Keep a relative vector check as well as coordinate tolerances;
                # the essential float64 reference test retains 1e-12 tolerances.
                torch.testing.assert_close(details['output'], output, rtol=1e-5, atol=1e-6)
                relative_error = torch.linalg.vector_norm(details['output'] - output) / torch.linalg.vector_norm(output)
                if relative_error > 1e-6:
                    raise AssertionError('manual attention differs beyond float32 accumulation error')
                torch.testing.assert_close(details['probabilities'].sum(-1), torch.ones_like(details['entropy']), rtol=1e-6, atol=1e-6)
                details['actual_output'] = output
                attention[name] = {k: v.detach().numpy().copy() for k, v in details.items()}
            return hook

        if exchange:
            for name, module in [('fine', models['E'].fine_from_coarse), ('coarse', models['E'].coarse_from_fine)]:
                handles.append(module.register_forward_hook(capture_attention(name)))
                handles.append(getattr(models['E'], name + '_attention_norm').register_forward_pre_hook(capture_incoming(name)))
        # NPZ reload makes NCHW contiguous, whereas FrameSet returns channels-last
        # strides. Identical pixels can select a different convolution kernel.
        # Restore the original data path so saved latents must match bit for bit.
        rgb, targets = test.batch(raw['test_indices'])
        np.testing.assert_array_equal(rgb.numpy(), raw['rgb'])
        np.testing.assert_array_equal(targets.numpy(), raw['targets'])
        z = models['E'](rgb)
        for handle in handles:
            handle.remove()
        np.testing.assert_array_equal(z.fine.numpy(), raw['fine'])
        np.testing.assert_array_equal(z.coarse.numpy(), raw['coarse'])
        attention_metrics, arrays = {}, dict(**pca_arrays)
        for name, details in attention.items():
            probabilities, ent = details['probabilities'], details['entropy']
            keys = probabilities.shape[-1]
            np.testing.assert_allclose(ent.mean(1), raw[name + '_entropy'], rtol=1e-6, atol=1e-7)
            actual, uniform = details['actual_output'], details['uniform_output']
            attention_metrics[name] = dict(keys=keys, mean_entropy=float(ent.mean()),
                head_mean_entropy=ent.mean((0, 2)).tolist(), entropy_min=float(ent.min()), entropy_max=float(ent.max()),
                mean_effective_keys=float(np.exp(ent * np.log(keys)).mean()),
                entropy_of_head_mean=float(entropy(torch.from_numpy(probabilities.mean(1))).mean()),
                manual_output_max_abs_error=float(np.abs(actual - details['output']).max()),
                manual_output_relative_l2_error=float(np.linalg.norm(actual - details['output']) / np.linalg.norm(actual)),
                saved_entropy_max_abs_error=float(np.abs(ent.mean(1) - raw[name + '_entropy']).max()),
                uniform_relative_output_error=float(np.linalg.norm(uniform - actual) / np.linalg.norm(actual)),
                attention_to_incoming_norm_ratio=float(np.linalg.norm(actual) / np.linalg.norm(incoming[name])),
                incoming_variance=variance_parts(incoming[name]), output_variance=variance_parts(actual))
            for key, value in details.items():
                arrays[name + '_' + key] = value
            arrays[name + '_incoming'] = incoming[name]
        if file_hash(checkpoint) != before:
            raise RuntimeError('checkpoint changed')
        summaries[arm] = dict(checkpoint=str(checkpoint), checkpoint_sha256=before, source_states=str(raw_path),
            source_states_sha256=file_hash(raw_path), source_manifest_sha256=file_hash(manifest_path),
            train_indices=raw['train_pca_indices'].tolist(), test_indices=raw['test_indices'].tolist(),
            dataset_fingerprint=train.fingerprint, training_frames=len(bank), test_frames=len(raw['rgb']),
            joint_pca_top3_variance_fraction=joint_fraction, pca=pca_metrics, attention=attention_metrics,
            checkpoint_unchanged=True, original_latents_exact=True)
        np.savez_compressed(OUT / f'{arm}_audit.npz', **arrays, **{'color_' + k: v for k, v in colors.items()})
        if arm == 'deeper':
            plot_pca(raw, colors)
            plot_attention(raw, attention)
        print(arm, json.dumps(dict(pca=pca_metrics, attention=attention_metrics)), flush=True)
    json_atomic(OUT / 'summary.json', dict(status='completed', kind='frozen visualization audit', seed=7107,
        optimizer_updates=0, versions=dict(torch=torch.__version__, numpy=np.__version__),
        code={p:file_hash(p) for p in [__file__, 'world_model/curriculum/encoder_visual_audit.py']}, arms=summaries))
    d = summaries['deeper']
    narrative = ('## Encoder PCA and attention audit\n\n'
        'Four frozen first-seed encoders; the original 256 training PCA frames and six fixed test frames. '
        'Original arrays and checkpoints are unchanged. Manual attention outputs and saved per-query entropy maps agree. '
        f'The deeper/on joint PCA displays {100*d["joint_pca_top3_variance_fraction"]:.1f}% of total token variance. '
        f'Image-to-image mean differences account for {100*d["pca"]["coarse"]["training_variance"]["image_mean_fraction"]:.1f}% '
        'of coarse training-token variance. Separate-scale and per-image-centered views expose different variation; '
        'the latter deliberately removes image-level information. Colors are unrelated across separate PCA fits. '
        f'Fine queries over 64 coarse keys have mean normalized entropy {d["attention"]["fine"]["mean_entropy"]:.6f}; '
        f'coarse queries over 256 fine keys have {d["attention"]["coarse"]["mean_entropy"]:.6f}. '
        'These measure routing concentration, not utility, certainty, or correctness. '
        'Per-head values, effective key counts, uniform-output differences and exact output checks are in the raw summary. '
        'No new architecture, optimizer updates, trained fusion ablations, or control evaluations ran.')
    sources = [p for p in OUT.iterdir() if p.suffix in ('.npz', '.json', '.png')]
    panels = [dict(file=str(OUT / 'deeper_pca_comparison.png'), title='Encoder PCA audit',
        caption='First three of the original six fixed test views, deeper/on seed7107. Each new PCA is trained on the same 256 training images. Per-image centering is only an inspection transform.', embed=True),
        dict(file=str(OUT / 'deeper_attention_heads.png'), title='Individual cross-attention heads',
        caption='First original test frame, pusher-cell query. Probability divided by uniform probability; clipped display0–4, exact unclipped arrays retained.', embed=False)]
    analyze(ROOT, OUT, 'Frozen encoder PCA and attention audit',
        dict(encoders=4, training_frames_per_encoder=256, test_frames_per_encoder=6, optimizer_updates=0),
        narrative, sources, panels)


if __name__ == '__main__':
    # Preserve the original encoder study's scope. A global runs/ refresh mixes
    # historical ledgers whose source paths are relative to their own studies.
    from scripts.run_encoder_study import refresh
    try:
        main()
    finally:
        artifact, html, receipt = refresh()
        verification = receipt.get('stages', {}).get('verification', 'unknown')
        print(f'Dashboard: {html}\nArtifact: {artifact}\nVerification: {verification}')
        if verification != 'passed':
            raise RuntimeError('dashboard browser verification remains pending')
