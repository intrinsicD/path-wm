"""Publish compact audit evidence from the completed immutable numeric records."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_encoder_study import refresh
from world_model.curriculum.pose_accessibility import analyze

ROOT = Path('runs/encoder_study_2026-09-08')
OUT = ROOT / 'evaluation/visual_audit'


def publish():
    summary = json.loads((OUT / 'summary.json').read_text())
    original = dict(np.load(ROOT / 'evaluation/internals/deeper_states.npz'))
    audit = dict(np.load(OUT / 'deeper_audit.npz'))
    rgb = original['rgb'].transpose(0, 2, 3, 1)
    fine = audit['color_joint_fine'].reshape(-1, 16, 16, 3)
    coarse = audit['color_joint_coarse'].reshape(-1, 8, 8, 3)
    separate = audit['color_coarse'].reshape(-1, 8, 8, 3)
    centered = audit['color_coarse_centered'].reshape(-1, 8, 8, 3)
    # Full comparison stays available with shorter, unclipped row labels.
    fig, axes = plt.subplots(5, 3, figsize=(7.5, 9), layout='constrained')
    for row, (label, values) in enumerate([('Input', rgb), ('Fine: joint PCA', fine),
            ('Coarse: joint PCA', coarse), ('Coarse: own PCA', separate), ('Coarse: centered PCA', centered)]):
        for col in range(3):
            axes[row, col].imshow(values[col], interpolation='nearest')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f'Test row {original["test_indices"][col]}', fontsize=10)
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.suptitle('Deeper/on · seed 7107 · PCA inspection alternatives\nTraining-fixed fits and color bounds; independent rows do not share color meanings', fontsize=10)
    fig.savefig(OUT / 'deeper_pca_comparison.png', dpi=110)
    plt.close(fig)
    # One compact fixed example fits beside all existing formal-study evidence.
    fig, axes = plt.subplots(1, 3, figsize=(6, 2.3), layout='constrained')
    for ax, title, values in zip(axes, ['Input', 'Coarse: joint PCA', 'Coarse: centered PCA'], [rgb, coarse, centered]):
        ax.imshow(values[0], interpolation='nearest'); ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Deeper/on · seed 7107 · first fixed test row 1614\nSeparate color bases; centering removes the image mean', fontsize=9)
    fig.savefig(OUT / 'deeper_pca_compact.png', dpi=100)
    plt.close(fig)
    d = summary['arms']['deeper']
    heads = ', '.join(f'{v:.3f}' for v in d['attention']['coarse']['head_mean_entropy'])
    narrative = ('## Encoder PCA and attention audit\n\n'
        'Four frozen first-seed encoders; original256 training PCA frames and six fixed test frames. '
        'No optimizer updates. Checkpoints are unchanged; restored source strides reproduce saved latents exactly. '
        'Independent attention calculations reproduce every saved per-query entropy map exactly and match SDPA outputs within float32 roundoff. '
        f'The deeper/on joint PCA retains {100*d["joint_pca_top3_variance_fraction"]:.1f}% of pooled variance. '
        f'Image means explain {100*d["pca"]["coarse"]["training_variance"]["image_mean_fraction"]:.2f}% of coarse training-token variance '
        f'versus {100*d["pca"]["fine"]["training_variance"]["image_mean_fraction"]:.2f}% for fine. '
        'Separate-scale and image-centered PCA still show strong coarse spatial/boundary structure; nicer PCA does not establish local semantic information.\n\n'
        f'Fine queries over64 coarse keys: mean normalized entropy {d["attention"]["fine"]["mean_entropy"]:.6f}; '
        f'coarse queries over256 fine keys: {d["attention"]["coarse"]["mean_entropy"]:.6f}. '
        f'Coarse per-head means are {heads}; the mean hides a selective third head. '
        f'Of the coarse attention-output variance on the six frames, {100*d["attention"]["coarse"]["output_variance"]["image_mean_fraction"]:.2f}% '
        'is variation between image means, consistent with a shared image summary. '
        'Entropy describes routing concentration, not quality or certainty. Uniform routing changes this trained module output, '
        'but these local differences are not end-to-end intervention results or evidence of a trained replacement.\n\n'
        'The dashboard embeds the first declared test example to preserve its size limit. Full three-view PCA and individual-head '
        'panels, all six-frame arrays and exact metrics remain beside the audit summary. No architecture or training objective changed.')
    panels = [dict(file=str(OUT / 'deeper_pca_compact.png'), title='Coarse PCA audit: first fixed example',
        caption='Deeper/on seed7107, test row1614. Training-only PCA; new centered view removes image mean. Full three-view comparison and all six numerical cases retained.', embed=True)]
    sources = [p for p in OUT.iterdir() if p.suffix in ('.npz', '.json', '.png') and p.name != 'curriculum_analysis.json']
    analyze(ROOT, OUT, 'Frozen encoder PCA and attention audit',
        dict(encoders=4, training_frames_per_encoder=256, test_frames_per_encoder=6, optimizer_updates=0),
        narrative, sources, panels)


if __name__ == '__main__':
    publish()
    _, html, receipt = refresh()
    print('Dashboard:', html, 'Verification:', receipt.get('stages', {}).get('verification'))
    if receipt.get('stages', {}).get('verification') != 'passed':
        raise RuntimeError('dashboard verification incomplete')
