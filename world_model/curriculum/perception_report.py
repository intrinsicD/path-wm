"""Build a source-backed technical report with the existing portable HTML reader."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo
import numpy as np
import torch
from .perception_cache import ROOT
from .perception_summary import reconcile,read,lines
from .perception_localization import map_diagnostics
from .data import file_hash
from world_model.pusht.checkpoints import json_atomic
from viewer.scientific_preview import image_data_uri

PROJECT=Path(__file__).resolve().parents[2]
SOURCE='overnight_evidence'


def baseline_maps():
    rows=[]
    for seed in (9107,9108,9109):
        for arm in ('cnn','vit'):
            out=ROOT/'fresh/evaluations'/f'seed_{seed}'/arm
            raw=dict(np.load(out/'fresh_errors.npz')); target=raw['targets']
            with torch.no_grad():
                d=map_diagnostics(torch.from_numpy(raw['locations']),torch.from_numpy(target[:,:4]).float().reshape(-1,2,2))
            edge=np.minimum(target[:,:2],1-target[:,:2]).min(1)*512<48
            rows.append(dict(arm=arm,seed=seed,objective='P1 coordinate/orientation MSE',
                entropy=float(d['location_entropy'].mean()),support_mass=float(d['target_support_mass'].mean()),
                kl=float(d['location_kl'].mean()),boundary_cases=int(edge.sum()),
                boundary_pass=float((raw['case_q'][edge]<=1).mean()),interior_pass=float((raw['case_q'][~edge]<=1).mean())))
    return rows


def geometry_failures():
    rows=[]
    for seed in (9107,9108,9109):
        for arm in ('cnn','vit'):
            baseline=ROOT/f'fresh/evaluations/seed_{seed}/{arm}/fresh_errors.npz'
            reference=dict(np.load(baseline))
            for objective,path in [('P1 MSE',baseline),('MSE + location KL',ROOT/f'localization/seed_{seed}/{arm}/fresh_errors.npz')]:
                if not path.exists(): continue
                raw=dict(np.load(path))
                if not np.array_equal(raw['indices'],reference['indices']) or not np.array_equal(raw['targets'],reference['targets']):
                    raise AssertionError('geometry stress populations differ')
                errors=raw['position_abs_error']; failures=(errors>8).sum(0)
                rows.append(dict(arm=arm,seed=seed,objective=objective,frames=len(errors),
                    pusher_x=int(failures[0]),pusher_y=int(failures[1]),block_x=int(failures[2]),block_y=int(failures[3]),
                    orientation=int((raw['angle_abs_error_deg']>10).sum()),any_component=int((raw['case_q']>1).sum()),
                    case_q_p95=float(np.percentile(raw['case_q'],95)),case_q_p99=float(np.percentile(raw['case_q'],99)),
                    case_q_max=float(raw['case_q'].max()),worst_case=int(raw['indices'][raw['case_q'].argmax()])))
    return rows


def build():
    report,dashboard=reconcile(); now=report['generated']; complete=report['status']=='complete'
    out=ROOT/'report'; title='Encoder and decoder experiments · 8–9 September 2026'
    datasets=copy.deepcopy(report['datasets']); blocks=[]; charts=[]; tables=[]; prose=[]
    available_charts={c['id']:c for c in dashboard['manifest']['charts']}
    available_tables={t['id']:t for t in dashboard['manifest']['tables']}
    def md(ident,body):
        blocks.append(dict(id=ident,type='markdown',body=body,sourceId=SOURCE)); prose.append(body)
    def chart(ident):
        if ident not in available_charts: return
        item=copy.deepcopy(available_charts[ident]); item['sourceId']=SOURCE; charts.append(item)
        blocks.append(dict(id='chart_'+ident,type='chart',chartId=ident,layout='full'))
    def table(ident):
        if ident not in available_tables: return
        item=copy.deepcopy(available_tables[ident]); item['sourceId']=SOURCE; tables.append(item)
        blocks.append(dict(id='table_'+ident,type='table',tableId=ident,layout='full'))
    def custom_table(ident,title,rows,columns,subtitle):
        datasets[ident]=[{field:row.get(field) for field,_ in columns} for row in rows]
        tables.append(dict(id=ident,dataset=ident,sourceId=SOURCE,title=title,subtitle=subtitle,
            defaultSort=dict(field=columns[0][0],direction='asc'),layout='full',density='dense',
            columns=[dict(field=f,label=l,type='text') for f,l in columns]))
        blocks.append(dict(id='table_'+ident,type='table',tableId=ident,layout='full'))
    def figure(ident,relative,caption):
        path=ROOT/relative
        if not path.exists(): return
        encoded=image_data_uri(path)
        blocks.append(dict(id=ident,type='html',layout='full',sourceId=SOURCE,
            body=f'<figure><img src="{encoded}" alt="{caption}" style="width:100%;height:auto"><figcaption>{caption}</figcaption></figure>'))
        prose.append(f'![{caption}]({path.resolve()})')
    def mean(dataset,arm,metric):
        value=report['descriptive_groups'].get(dataset,{}).get(arm)
        return None if value is None else value['metrics'][metric]['mean']
    def metric(dataset,arm,field,percent=False):
        value=mean(dataset,arm,field)
        return 'pending' if value is None else (f'{100*value:.1f}%' if percent else f'{value:.4g}')

    c=report['counts']
    local_time=datetime.fromisoformat(now).astimezone(ZoneInfo('Europe/Berlin')).strftime('%d %B %Y, %H:%M')
    md('title',f'# {title}\n\n'+('**Completed comparison.**' if complete else '**Partial report: training/evaluation still in progress.**')+
        f' As of {local_time} Berlin. {c["completed_vision_fits"]}/{c["expected_vision_fits"]} vision/geometry fits and '
        f'{c["completed_semantic_fits"]}/{c["expected_semantic_fits"]} category probes have completed and reconciled. '
        'Albedo is excluded. This report evaluates perception and decoder prototypes; new dynamics and control have not been trained.')
    md('summary','## Technical summary\n\n'
        '**Use a pretrained contextual encoder with an available local image path and output-specific decoders as the next integration candidate.** '
        'Keep the compact CNN as a speed reference. The experiments favor an asymmetric encoder/decoder package; they do not establish a universal world-model architecture.\n\n'
        'The pretrained ViT is much stronger on the tested foreground and category readouts. Its pixel-reconstruction weakness can be substantially reduced '
        'without changing that encoder: early patch features improve RGB and segmentation in all three paired seeds versus final-token inputs alone. '
        'Separate RGB/mask trunks then reduce mean COCO RGB error by7.42× versus the shared early-input trunk, while lowering foreground IoU by1.02 percentage points. '
        'Output sharing therefore remains a measured tradeoff. RGB/mask FiLM does not improve both outputs consistently, so it is not recommended by default.\n\n'
        'Adding two convolutional or transformer blocks per CNN scale produces no consistent useful gain under the matched continuation budget. '
        'Location-distribution supervision improves fresh-case tolerance pass from76.6% to84.9% for CNN and80.5% to85.3% for ViT. '
        'The CNN acquires worse rare errors; the ViT improves both95th- and99th-percentile errors in all three seeds but still has large outliers. '
        'The next priority is geometry coverage and reliable temporal prediction, not additional scale count alone. '
        'Pretraining, size and preprocessing confound CNN-versus-ViT comparisons; these results do not justify the rule that convolutions are spatial and transformers semantic.'
        if complete else
        '## Technical summary\n\nA versatile perception package should expose local spatial evidence and contextual features to typed consumers. '
        'The comparisons below test encoder depth, decoder input access, task conditioning, output sharing and geometry objectives. '
        'This partial report does not select a final architecture until every declared unit is reconciled.')
    md('definitions','## Populations, units and what the metrics establish\n\n'
        'Three paired seeds (9107–9109) vary new head/module initialization and sampling. They do not resample encoder pretraining. '
        'The CNN is one preserved deeper/exchange checkpoint; DINOv2-S/14 uses pinned official weights/source and RGB64→224 preprocessing. '
        'Every formal vision fit receives 4,000 updates. Dense/encoder fits use 32 COCO plus 32 PushT frames per update; '
        'the objective-only geometry fits consume 32 PushT frames, preserving the original sampler stream.\n\n'
        'COCO uses 4,096 training, 512 validation and 512 test crops. The mask is the union of visible non-crowd foreground, '
        'with crowd-only pixels ignored; this does not test arbitrary object-query segmentation. Category probes retain 78 classes from training support '
        'and exclude unknown crowd labels. AP is recomputed from raw scores with tied thresholds grouped.\n\n'
        'PushT CCHI has 20,493 training, 2,651 validation and 2,506 test frames in disjoint initial-configuration groups. '
        'These grouped holdouts have been used by earlier design iterations and remain exploratory. A separate 512-case simulator reset cohort tests broader geometry. '
        'All cases are retained, including the declared near matches; renderer calibration uses 40 source/render pairs.\n\n'
        '**q** is the maximum of four coordinate MAEs divided by 8 world units and wrapped orientation MAE divided by 10°. '
        'Mean q≤1 is the existing perception gate. **Per-case pass** requires every coordinate error≤8 and orientation error≤10° in that individual frame. '
        'It is stricter and is not control success. Positions refer to pusher coordinates and the T-body origin, not necessarily its shape centroid. '
        '**RGB MSE** uses images in [0,1]; lower is better. **Mask IoU** is the mean per-image foreground intersection/union over valid pixels; higher is better. '
        '**Category AP** measures accessibility through the declared pooled classifier. No composite architecture score or significance threshold is invented.')
    md('earlier_recovery','The earlier frozen-encoder recovery test reduced COCO RGB error from0.272195 to0.006105, '
        'close to the original warmup value0.006072, while worsening PushT reconstruction by8.72×. '
        'That showed substantial information was still decodable after adaptation, alongside a decoder/domain tradeoff. '
        f'These are [historical recovery results]({PROJECT / "docs/decoder-recovery-results-2026-09-08.md"}); '
        'they motivated mixed-domain decoding here and are not pooled with the new architecture comparisons.')
    md('p1','## Frozen encoders with independent typed readouts\n\n'
        'The CNN reference has16×16 and8×8 grids of64 channels, two residual convolutional blocks per branch, '
        'learned position/scale signals and bidirectional four-head cross-scale exchange followed by token MLPs. '
        'Its coarse branch starts from the shared stem before fine-branch residual refinement. '
        'The pinned ViT uses twelve transformer blocks and384-channel final patch features; its8×8 grid is derived by pooling the16×16 final grid. '
        'That pooling adds a spatial view, not another independently processed encoder stage.\n\n'
        f'Across the three head seeds, CNN/ViT mean mask IoU is {metric("perception_package","cnn","mask_iou")}/'
        f'{metric("perception_package","vit","mask_iou")}, while COCO reconstruction MSE is '
        f'{metric("perception_package","cnn","coco_mse")}/{metric("perception_package","vit","coco_mse")}. '
        'The generic-output ranking therefore depends on the output being requested. Independent RGB, mask and spatial pose heads prevent direct head-gradient interference. '
        'All three heads use the one snapshot selected by minimum validation q, with earliest exact ties. The unused endpoint remains recorded. '
        'This checkpoint rule may not optimize RGB or segmentation; it was fixed before fitting.')
    chart('perception_package_mask_iou'); chart('perception_package_coco_mse'); table('perception_package')
    md('semantics','## Category evidence and transfer\n\n'
        f'The frozen CNN/ViT category AP means are {metric("perception_semantics","cnn","test_ap")}/'
        f'{metric("perception_semantics","vit","test_ap")}, against a constant-score baseline of 0.03310. '
        'Continued CNN variants use separately fitted probes on their selected encoders. The same training-supported class list, '
        'pooled 128-wide classifier, paired sample streams, 2,000-update budget and maximum-validation-AP selector apply. '
        'Native feature widths and pretraining differ. A poor result from this pooled head does not prove that all semantic information is absent.')
    chart('perception_semantics'); table('perception_semantics')
    figure('generic_images','figures/readouts/coco_readouts.png',
        'Six fixed COCO test crops: RGB, reconstructions, foreground target and probabilities. Pink denotes ignored crowd-only pixels; every probability image uses the same 0–1 range.')
    md('fresh','## Fresh geometry exposes errors hidden by the mean gate\n\n'
        f'Frozen CNN/ViT mean per-case pass is {metric("perception_fresh","cnn","per_case_pass",True)}/'
        f'{metric("perception_fresh","vit","per_case_pass",True)}. '
        'The original ViT pose head has lower 95th-percentile case error in all three paired seeds. '
        'This supports testing generalization and error tails rather than treating an average readiness pass as reliable state estimation. '
        'The fresh reset population contains more edge configurations than demonstrations; it is a stress test, not an estimate of the deployment distribution.')
    chart('perception_fresh'); table('perception_fresh')
    figure('pose_maps','figures/fresh/fresh_pose_distributions_v2.png',
        'Four fixed fresh cases plus the worst CNN and ViT seed-9107 cases. Position distributions and orientation pools are decoder outputs, not encoder attention; probabilities share a log scale.')
    md('extensions','## More processing per encoder scale\n\n'
        'The matched comparison starts every arm from the same CNN and trains fresh typed heads. Joint continuation keeps its topology; '
        'the other arms add two zero-gated residual convolutional blocks or two transformer blocks per existing grid before cross-scale exchange. '
        'Source-encoder learning rate is 3e-5; heads and added blocks use 3e-4. The initial function is exactly preserved. '
        'The shared encoder receives mixed RGB, mask and pose objectives; all arms retain their q-selected checkpoint.\n\n'
        'Under these exact objectives and budgets, adding depth does not consistently improve held-out pose or reconstruction. '
        'This is a result about these zero-gated continuations, not a proof that deeper encoders cannot help. '
        'The pretrained ViT comparison tests a different package; it is not the control for local transformer insertion.')
    chart('perception_continuation_test_q'); chart('perception_continuation_mask_iou'); table('perception_continuation')
    custom_table('pose_group_balance','Grouped-test robustness and orientation output norms',
        report['datasets'].get('perception_package',[])+report['datasets'].get('perception_continuation',[]),
        [('study','Study'),('arm','Arm'),('seed','Seed'),('test_q','Frame-weighted q'),('group_balanced_q','Equal-group q'),
         ('groups','Test groups'),('angle_norm_mean','Mean sin/cos vector norm'),('near_zero_angle_count','Norm below1e-6 count')],
        'Group-balanced q weights each initial-configuration group equally before applying the same physical tolerances. It is descriptive; checkpoint selection still uses the declared frame-weighted validation q.')
    diagnostics=report['encoder_training_diagnostics']
    custom_table('encoder_gradients','Shared-encoder gradient diagnostics',diagnostics,
        [('arm','Arm'),('seed','Seed'),('generic_manipulation_gradient_norm_ratio_median','Generic/manipulation norm ratio'),
         ('domain_gradient_cosine_median','Median cosine'),('negative_gradient_cosine_fraction','Negative cosine fraction'),
         ('encoder_clip_fraction','Clipped update fraction')],
        'Audited updates from1,000 onward; generic is COCO RGB+mask, manipulation is PushT RGB+pose, before encoder clipping.')
    md('gradient_reading','The generic objective produces much larger encoder gradient norms than the manipulation objective. '
        'This is a measured optimization imbalance; small or mixed gradient cosines do not by themselves prove harmful transfer. '
        'Small residual gates likewise do not establish unused branches because branch weights may compensate. '
        'A future loss-balancing or ungated-training comparison would need its own matched control.')
    # Remaining sections are added by finish_sections so partial reports have the same reading path.
    finish_sections(report,md,chart,table,custom_table,figure,metric,mean)
    all_paths=[]
    for name in report['sources']:
        matches=[p for p in (PROJECT/name,PROJECT/'runs'/name) if p.is_file()]
        if not matches: raise FileNotFoundError('report input identity is unresolved: '+name)
        all_paths.append(matches[0])
    all_paths += [Path(__file__),Path('world_model/curriculum/perception_summary.py'),Path('viewer/perception.py')]
    all_paths += sorted(Path('docs').glob('perception-*-protocol-2026-09-*.md'))
    all_paths += [Path('docs/decoder-recovery-results-2026-09-08.md'),Path('viewer/scientific_preview.py')]
    all_paths += [Path('docs/dashboard-source-inventory-repair-2026-09-09.md'),
        ROOT/'report/pytest_final.xml',ROOT/'report/pytest_final_v2.xml']
    all_paths += sorted((ROOT/'collaboration').glob('*.md'))+sorted((ROOT/'collaboration').glob('*.receipt.json'))
    all_paths += [ROOT/'figures/pca/feature_pca_v2.png',ROOT/'figures/readouts/coco_readouts.png',ROOT/'figures/fresh/fresh_pose_distributions_v2.png']
    identities={str(p.resolve().relative_to(PROJECT)):file_hash(p) for p in all_paths if p.is_file()}
    sources=[dict(id=SOURCE,label='Reconciled local experiments, protocols and raw predictions',path=str(out.resolve().relative_to(PROJECT)/('reconciled.json' if complete else 'reconciled_partial.json')),
        query=dict(language='python',executed_at=now,description='Hash-verified analysis records are reconciled against raw pose predictions, per-image mask/RGB metrics, category scores, validation selectors, actual update counts and paired sampling streams. Charts retain individual seeds. Descriptive means/ranges do not establish confidence intervals.',
            input_files=sorted(identities),input_sha256=identities,transformation='world_model/curriculum/perception_summary.py; viewer/perception.py; world_model/curriculum/perception_report.py',
            filters=['Development fits excluded from formal comparisons','Stopped or missing arms remain explicitly pending','Fresh cohort includes all512 declared cases'],
            metric_definitions=dict(q='max(max four coordinate MAE/8worldunits, wrapped angle MAE/10degrees)',per_case_pass='fraction satisfying every8-unit coordinate and10-degree angle tolerance',RGB_MSE='mean squared difference in[0,1]',mask_IoU='mean per-image intersection/union over valid pixels',category_AP='macro AP over training-supported classes with known labels')))]
    # The canonical reader requires actual SQL for quantitative widget sources.
    # Python performs scientific reconciliation; these executed projections expose
    # exactly the verified rows without pretending the experiments live in a DB.
    binding_path=out/('widget_rows.json' if complete else 'partial_widget_rows.json')
    json_atomic(binding_path,datasets); queries={}
    with sqlite3.connect(':memory:') as connection:
        connection.row_factory=sqlite3.Row
        for name in sorted({w['dataset'] for w in [*charts,*tables]}):
            original=datasets[name]; fields=sorted({key for row in original for key in row})
            if not fields: continue
            if not all(re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*',field) for field in fields): raise ValueError('unsafe widget SQL field')
            sql='SELECT '+', '.join(f'json_extract(value,\'$.{field}\') AS "{field}"' for field in fields)+' FROM json_each(:verified_rows) ORDER BY CAST(key AS INTEGER);'
            selected=[dict(row) for row in connection.execute(sql,{'verified_rows':json.dumps(original)})]
            if selected!=[{field:row.get(field) for field in fields} for row in original]: raise AssertionError('SQL widget projection changed verified values')
            datasets[name]=selected; queries[name]=sql
            source_id='query_'+name
            for widget in [*charts,*tables]:
                if widget['dataset']==name: widget['sourceId']=source_id
            sources.append(dict(id=source_id,label='Verified rows · '+name,path=str(binding_path.resolve().relative_to(PROJECT)),
                query=dict(engine='sqlite',sql=sql,executed_at=now,
                    description='Executed ordered projection over the named dataset in the preserved JSON binding. Scientific metrics were computed and validated in Python first; this query preserves every widget value.',
                    input_files=[str(binding_path.resolve().relative_to(PROJECT))],input_sha256=file_hash(binding_path),
                    binding_dataset=name,tables_used=['json_each(:verified_rows)'],transformation='world_model/curriculum/perception_report.py: build SQL widget projections')))
    json_atomic(out/('widget_queries.json' if complete else 'partial_widget_queries.json'),queries)
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,description='Measured perception packages, encoder depth, local decoder evidence, task conditioning and geometry objectives.',generatedAt=now,
        blocks=blocks,cards=[],charts=charts,tables=tables,filters=[],sources=sources),
        snapshot=dict(version=1,generatedAt=now,status='ready' if complete else 'partial',datasets=datasets),sources=sources)
    destination=out/('report.artifact.json' if complete else 'partial.artifact.json'); json_atomic(destination,artifact)
    (out/('report.md' if complete else 'partial.md')).write_text('\n\n'.join(prose)+'\n')
    print(json.dumps(dict(artifact=str(destination),status=report['status'],blocks=len(blocks),charts=len(charts),tables=len(tables))))
    return destination


def finish_sections(report,md,chart,table,custom_table,figure,metric,mean):
    md('decoders','## Local decoder inputs and task conditioning\n\n'
        'Every arm has a shared convolutional dense trunk and separate RGB/mask output layers. Three slots project to128 channels each: '
        'final16×16 ViT tokens, their8×8 average pool, and one varying input. Late duplicates the final tokens; early supplies patch embeddings '
        'before transformer blocks/positions; raw supplies disjoint4×4 RGB patches. Conditioned uses the same early inputs with RGB/mask FiLM '
        'at every decoder resolution. All frozen final features retain384 channels before per-consumer projection.\n\n'
        'Each token is normalized while preserving its original mean and scale as two additional values. This makes the pre-projection pack invertible. '
        'Raw input has fewer projection parameters and different receptive support than the upscaled patch embedding. '
        'Early-versus-conditioned isolates the task signal; typed final output layers already distinguish tasks in every arm. '
        'This tests task-dependent shared computation, not whether the model can identify its output type.\n\n'
        'All four arms use fixed4,000-update endpoints, paired examples, identical losses and a shared-decoder optimizer. '
        'Validation is monitoring only. Primary outcome is COCO foreground IoU; RGB error in both domains and compute remain separate tradeoffs. '
        'Earlier independent P1 trunks use another checkpoint selector and are contextual references, not a matched shared-versus-independent experiment.')
    chart('perception_decoders_mask_iou'); chart('perception_decoders_coco_mse'); chart('perception_decoders_pusht_mse'); table('perception_decoders')
    figure('decoder_rgb','figures/decoders/rgb_comparison.png','Fixed first-six COCO RGB outputs from all four seed9107 endpoints. Same present observation and display range.')
    figure('decoder_mask','figures/decoders/mask_comparison.png','Fixed first-six COCO foreground outputs from all four seed9107 endpoints. Probabilities share the same0–1 range; ignored crowd pixels are pink.')
    rows=report['datasets'].get('perception_decoders',[])
    gradient_rows=[]
    for row in rows:
        out=ROOT/'decoders'/f'seed_{row["seed"]}'/row['arm']; training=lines(out/'training.jsonl')
        audited=[r for r in training if r['step']>=1000 and 'rgb_mask_gradient_cosine' in r]
        if audited:
            gradient_rows.append(dict(arm=row['arm'],seed=row['seed'],audited_updates=len(audited),
                cosine_median=float(np.median([r['rgb_mask_gradient_cosine'] for r in audited])),
                negative_fraction=float(np.mean([r['rgb_mask_gradient_cosine']<0 for r in audited])),
                mask_rgb_norm_ratio=float(np.median([r['mask_shared_grad_norm']/max(r['rgb_shared_grad_norm'],1e-20) for r in audited])),
                clip_fraction=row['global_clip_fraction']))
    if len(rows)==12:
        lookup={(r['arm'],r['seed']):r for r in rows}; pairs=[]
        for baseline,candidate in [('late','early'),('early','raw'),('early','conditioned')]:
            for seed in (9107,9108,9109):
                a,b=lookup[baseline,seed],lookup[candidate,seed]
                pairs.append(dict(comparison=f'{candidate} minus {baseline}',seed=seed,
                    mask_iou_delta=b['mask_iou']-a['mask_iou'],coco_mse_ratio=b['coco_mse']/a['coco_mse'],
                    pusht_mse_ratio=b['pusht_mse']/a['pusht_mse'],both_time_ratio=b['both_ms_per_image']/a['both_ms_per_image']))
        custom_table('decoder_pairs','Paired decoder changes',pairs,
            [('comparison','Comparison'),('seed','Seed'),('mask_iou_delta','IoU change'),('coco_mse_ratio','COCO MSE ratio'),
             ('pusht_mse_ratio','PushT MSE ratio'),('both_time_ratio','Both-output timing ratio')],
            'Positive IoU change and MSE ratios below1 favor the candidate. Ratios compare matched seeds; measured timing includes prepared decoder inputs only.')
        md('decoder_interpretation',f'Mean COCO RGB errors for late/early/raw/conditioned are '
            f'{metric("perception_decoders","late","coco_mse")}, {metric("perception_decoders","early","coco_mse")}, '
            f'{metric("perception_decoders","raw","coco_mse")}, {metric("perception_decoders","conditioned","coco_mse")}. '
            'The paired table shows whether each change improves masks as well as pixels, and whether the effect repeats across seeds. '
            'Prepared-input inference timing excludes the frozen encoder and preprocessing. Unconditioned outputs reuse one trunk evaluation; '
            'task-modulated outputs require two when both are requested. A single requested output still uses one pass; '
            'individual RGB/mask timings remain in each raw evaluation. Training uses separate task forwards in every arm, preserving the declared comparison.')
        early_wins=sum(lookup['early',s]['coco_mse']<lookup['late',s]['coco_mse'] and lookup['early',s]['mask_iou']>lookup['late',s]['mask_iou'] for s in (9107,9108,9109))
        film_rgb_wins=sum(lookup['conditioned',s]['coco_mse']<lookup['early',s]['coco_mse'] for s in (9107,9108,9109))
        film_mask_losses=sum(lookup['conditioned',s]['mask_iou']<lookup['early',s]['mask_iou'] for s in (9107,9108,9109))
        md('decoder_decision',f'Early inputs improve both COCO RGB error and mask IoU in {early_wins}/3 paired seeds versus late inputs. '
            f'Task FiLM lowers COCO RGB error in {film_rgb_wins}/3 seeds but lowers mask IoU in {film_mask_losses}/3 seeds versus the same unconditioned early decoder. '
            'The task signal therefore has a measured tradeoff, rather than a general benefit across outputs. '
            'The raw-source package does not match early-input RGB accuracy under this budget; preprocessing support and projection width differ. '
            'These results support supplying local evidence and do not justify enabling task FiLM by default.')
    else:
        md('decoder_interpretation','Decoder results are pending. No local-input or task-conditioning winner is selected in this partial report.')
    if gradient_rows:
        custom_table('decoder_gradients','Shared-trunk gradient diagnostics',gradient_rows,
            [('arm','Arm'),('seed','Seed'),('audited_updates','Audited late updates'),('cosine_median','Median gradient cosine'),
             ('negative_fraction','Negative cosine fraction'),('mask_rgb_norm_ratio','Mask / RGB gradient norm'),('clip_fraction','Global clipped fraction')],
            'Audited updates1000 onward; shared parameters only. COCO mask BCE versus0.5×COCO RGB MSE, before PushT RGB gradients and clipping. These do not measure total domain conflict or prove a cause of quality differences.')
    md('split_decoders','## Separate RGB and mask trunks\n\n'
        'The adaptive D3 comparison keeps the early/final/pooled inputs, exact initial per-output functions, three paired seeds, '
        'losses, examples and fixed4,000-update endpoints. It gives each output its own complete dense trunk. '
        'This changes parameter ownership and cross-domain gradient transfer; the shared global clipping coefficient still couples branch scales. '
        'It is a sharing/capacity comparison, not a parameter-matched test or fully independent optimization. '
        'Both training paths perform three task passes per update. Separate trunks use nearly twice the parameters and optimizer storage, '
        'and two passes when both outputs are requested. GPU initial-function agreement is checked on actual inputs before each formal fit.')
    table('perception_independent')
    figure('split_output_panel','figures/split/decoder_comparison.png','Shared and separate trunks, same seed9107 endpoints and first six COCO test crops. The numerical three-seed comparison determines the tradeoff.')
    split_rows=report['datasets'].get('perception_independent',[])
    if split_rows and len(rows)==12:
        lookup={(r['arm'],r['seed']):r for r in rows}; pairs=[]
        for b in split_rows:
            for reference in ('early','conditioned'):
                a=lookup[reference,b['seed']]
                pairs.append(dict(comparison='split minus '+reference,seed=b['seed'],
                    mask_iou_delta=b['mask_iou']-a['mask_iou'],coco_mse_ratio=b['coco_mse']/a['coco_mse'],
                    pusht_mse_ratio=b['pusht_mse']/a['pusht_mse'],parameter_ratio=b['parameters']/a['parameters'],
                    both_time_ratio=b['both_ms_per_image']/a['both_ms_per_image'],
                    shared_clip_fraction=a['global_clip_fraction'],split_clip_fraction=b['global_clip_fraction']))
        custom_table('split_pairs','Separate trunks versus shared and conditioned counterparts',pairs,
            [('comparison','Comparison'),('seed','Seed'),('mask_iou_delta','IoU change'),('coco_mse_ratio','COCO MSE ratio'),
             ('pusht_mse_ratio','PushT MSE ratio'),('parameter_ratio','Parameter ratio'),('both_time_ratio','Both-output time ratio'),
             ('shared_clip_fraction','Reference clipped fraction'),('split_clip_fraction','Split clipped fraction')],
            'All ratios and differences use the same seed and endpoint budget. No composite winner score.')
        if len(split_rows)==3:
            rgb_ratio=mean('perception_decoders','early','coco_mse')/mean('perception_independent','split','coco_mse')
            pusht_ratio=mean('perception_decoders','early','pusht_mse')/mean('perception_independent','split','pusht_mse')
            iou_change=100*(mean('perception_independent','split','mask_iou')-mean('perception_decoders','early','mask_iou'))
            md('split_findings',f'The separate-trunk mean COCO RGB error is {metric("perception_independent","split","coco_mse")}, '
                f'{rgb_ratio:.2f}× lower than the shared early-input decoder. PushT RGB error is {pusht_ratio:.2f}× lower. '
                f'Mean mask IoU changes by {iou_change:+.2f} percentage points. The separate trunks therefore do not win the primary foreground endpoint; '
                'they offer a substantial appearance benefit at a modest foreground cost and higher decoder storage. '
                'The result supports output-specific computation, without establishing that sharing is always harmful or that the gain comes exclusively from gradient interference. '
                'The encoder can still supply common evidence to both outputs. A shared trunk with matched total capacity or partially shared task adapters remains an untested alternative.')
    md('decoder_reliance','## Which supplied inputs does the decoder use?\n\n'
        'A fixed derangement substitutes another image’s local slot or its paired final/coarse context while retaining the original RGB and mask targets. '
        'Conditioned endpoints also receive neutral or flipped FiLM context while retaining the requested output layer. '
        'RGB error ratios above1 and negative IoU changes indicate fitted dependence on the intervened input. '
        'Late has duplicated final features, so its first slot is not independent local evidence. '
        'The CPU normal conditions are checked against saved GPU endpoint metrics on the identical rows. '
        'These are distribution-shift interventions, not separately trained branch-removal controls.')
    table('perception_decoder_reliance')
    reliance=report['datasets'].get('perception_decoder_reliance',[])
    if len(reliance)==42:
        averages={condition:{field:float(np.mean([r[field] for r in reliance if r['arm']=='early' and r['condition']==condition]))
            for field in ('image_mse_ratio','mask_iou')} for condition in ('normal','donor_local','donor_context')}
        md('reliance_findings',f'For the early-input decoder, supplying another image’s local features increases RGB error by '
            f'{averages["donor_local"]["image_mse_ratio"]:.2f}× on average; replacing its final/coarse context increases it by '
            f'{averages["donor_context"]["image_mse_ratio"]:.2f}×. Mask IoU changes from '
            f'{averages["normal"]["mask_iou"]:.4f} normally to {averages["donor_local"]["mask_iou"]:.4f} with donor local inputs and '
            f'{averages["donor_context"]["mask_iou"]:.4f} with donor context. '
            'This fitted decoder depends much more on local evidence for appearance and contextual evidence for foreground output. '
            'Neutral/flipped task inputs also change conditioned outputs: the FiLM signal is used, even though it does not improve every output versus the unconditioned control.')
    md('geometry','## Geometry decoder: supervise the distribution as well as its mean\n\n'
        'The pose head takes a softmax over a16×16 location map and returns its expected coordinate. Coordinate MSE alone leaves the map shape underdetermined. '
        'The adaptive follow-up adds0.001×KL(target‖prediction) using bilinear weights on the four surrounding output-grid vertices. '
        'Their expected coordinate is exact even on image boundaries. No encoder, head topology, orientation pooling, augmentation or data change is introduced. '
        'The zero-weight development controls must reproduce the original P1 informative weights and outputs within2e-5 before formal fitting. '
        'Two proven spatial-softmax bias redundancies are reported separately; the original blanket-weight check and its failure are preserved.\n\n'
        'This follows the general idea of differentiable spatial expectation and distribution supervision in '
        '[DSNT](https://arxiv.org/abs/1801.07372) and [integral regression](https://arxiv.org/abs/1711.08229). '
        'The present sparse target, weight and boundary convention are specific experimental choices, not claims of a novel method. '
        'Probability mass outside the target support is discouraged through softmax normalization. The maps describe point supervision, '
        'not object extent or calibrated uncertainty. Orientation remains a separately checked output.\n\n'
        'A post-hoc six-pixel edge cut contains136/512 fresh cases (26.6%), versus3.30% of training frames. '
        'The cut motivated this follow-up and is not an independent confirmation dataset. No new boundary examples are added; '
        'a better objective cannot establish that missing training coverage has been repaired.')
    chart('perception_localization'); table('perception_localization')
    baselines=baseline_maps()
    custom_table('geometry_baseline_maps','Original P1 map and edge diagnostics',baselines,
        [('arm','Encoder'),('seed','Seed'),('entropy','Normalized map entropy'),('support_mass','Target support mass'),
         ('kl','Target-to-map KL'),('boundary_cases','Boundary cases'),('boundary_pass','Boundary case pass'),('interior_pass','Interior case pass')],
        'Computed from the preserved P1 fresh maps using exactly the same target basis and edge cut as the objective follow-up.')
    local=report['datasets'].get('perception_localization',[])
    if local:
        p1={(r['arm'],r['seed']):r for r in report['datasets']['perception_fresh'] if r['arm'] in ('cnn','vit')}
        paired=[dict(arm=r['arm'],seed=r['seed'],pass_delta_pp=100*(r['fresh_pass']-p1[r['arm'],r['seed']]['per_case_pass']),
            q_ratio=r['fresh_q']/p1[r['arm'],r['seed']]['q'],angle_delta_deg=r['angle_mae_deg']-p1[r['arm'],r['seed']]['angle_mae_deg']) for r in local]
        custom_table('geometry_pairs','Geometry objective versus its paired original P1 head',paired,
            [('arm','Encoder'),('seed','Seed'),('pass_delta_pp','Fresh pass change, pp'),('q_ratio','Mean-q ratio'),('angle_delta_deg','Angle error change°')],
            'Same architecture, frozen representation, initial head and PushT examples. Each arm keeps its own predeclared minimum-validation-q snapshot.')
        loss_rows=[]
        for row in local:
            out=ROOT/'localization'/f'seed_{row["seed"]}'/row['arm']; train=lines(out/'training.jsonl')
            for item in (train[0],train[-1]):
                loss_rows.append(dict(arm=row['arm'],seed=row['seed'],step=item['step'],pose_mse=item['pose_mse'],
                    weighted_kl=.001*item['location_kl'],ratio=.001*item['location_kl']/max(item['pose_mse'],1e-20)))
        custom_table('geometry_losses','Objective magnitudes at the first and last updates',loss_rows,
            [('arm','Encoder'),('seed','Seed'),('step','Update'),('pose_mse','Coordinate/orientation MSE'),('weighted_kl','0.001×KL'),('ratio','Weighted KL / MSE')],
            'These are sampled training-batch objective magnitudes, not gradient-norm ratios or held-out accuracy.')
    if len(local)==6:
        md('geometry_interpretation',f'All six paired runs improve the fresh per-case tolerance pass rate. CNN mean pass changes from '
            f'{metric("perception_fresh","cnn","per_case_pass",True)} to {metric("perception_localization","cnn","fresh_pass",True)}, '
            f'and ViT from {metric("perception_fresh","vit","per_case_pass",True)} to {metric("perception_localization","vit","fresh_pass",True)}. '
            'This does not make the objective a universal geometry fix: CNN fresh mean q,99th-percentile error and maximum error all worsen in every seed. '
            'ViT fresh mean q and95th/99th-percentile errors improve in every seed; orientation error is mixed and two seeds have a worse maximum. '
            'I would retain the ViT objective variant for the next coverage test, while keeping every failure visible.\n\n'
            'The fixed case86 panels show why an expected coordinate can fail: the pusher map has probability near the pusher and another region near the body, '
            'so its mean falls between them. That visible multimodality explains the readout displacement in this case; its upstream cause remains a hypothesis. '
            'Sparse edge coverage and object/scene correlations deserve a controlled test. Simply choosing the largest16×16 cell would sacrifice subpixel precision '
            'and is not a tested replacement. Map entropy is not a calibrated confidence score.')
    else:
        md('geometry_interpretation','Compare coordinate and orientation accuracy jointly with entropy and target-support mass. '
            'Sharper maps alone are not success. All error tails and seed differences remain visible while the comparison completes.')
    custom_table('geometry_component_failures','Which quantities exceed the per-case tolerance?',geometry_failures(),
        [('arm','Encoder'),('seed','Seed'),('objective','Objective'),('frames','Fresh cases'),
         ('pusher_x','Pusher x'),('pusher_y','Pusher y'),('block_x','Body x'),('block_y','Body y'),
         ('orientation','Orientation'),('any_component','Any failure')],
        'Counts above8 world units per coordinate or10 degrees orientation on the same512 cases. Component failures overlap and must not be summed as distinct failed cases.')
    custom_table('geometry_extremes','Typical tail and rare extreme errors',geometry_failures(),
        [('arm','Encoder'),('seed','Seed'),('objective','Objective'),('case_q_p95','95th percentile q'),
         ('case_q_p99','99th percentile q'),('case_q_max','Maximum case q'),('worst_case','Worst case index')],
        'Post-hoc descriptive tail audit from every saved case. More cases within tolerance can coexist with a worse extreme error. No new gate or model selector is introduced.')
    figure('cnn_location_maps','figures/localization/cnn_location_comparison.png','CNN: original and distribution-supervised pose heads on the same predefined first-seed cases. Common probability scale, target map positions in white.')
    figure('vit_location_maps','figures/localization/vit_location_comparison.png','ViT: original and distribution-supervised pose heads on the same predefined first-seed cases. Case q measures coordinate and orientation tolerance jointly.')
    md('internals','## Internal features and attention: what the pictures actually show\n\n'
        'PCA fits use128 fixed training images, balanced across COCO and PushT, independently for each representation and scale. '
        'Colors use training2nd/98th percentiles and deterministic component signs. Different bases have no common semantic color meaning. '
        'Global coarse PCA can emphasize image/domain means and shared positional structure; image-centered PCA exposes within-image variation. '
        'A visually strange coarse map therefore is not evidence by itself of failed features.')
    figure('feature_pca','figures/pca/feature_pca_v2.png',
        'Train-fitted feature PCA on fixed test crops. Global and image-centered coarse views answer different questions; the ViT coarse grid is average pooled from final tokens.')
    chart('perception_attention_entropy'); table('perception_attention_entropy')
    md('attention_interpretation','Entropy is computed over key probabilities separately for each head before averaging queries and images, '
        'and normalized by log(number of keys). Fine-from-coarse heads are almost uniform; coarse-from-fine contains one selective head '
        'with normalized entropy about0.328 while its other heads are near0.94–0.96. Averaging heads would conceal this difference. '
        'The explicit attention implementation agrees with scaled-dot-product attention to below1e-6 maximum absolute difference on the checked population.')
    chart('perception_attention_directional'); table('perception_attention_directional')
    md('attention_causal_scope','Replacing only fine-from-coarse weights with uniform weights leaves pose q close to baseline, '
        'but removing that branch degrades it sharply. Replacing coarse-from-fine attention with uniform weights also sharply degrades pose. '
        'This is consistent with broad context broadcast toward fine features and selective aggregation toward coarse features in this fitted system. '
        'These interventions establish reliance after training; they do not show that an independently retrained pooling or attention-free model would perform worse. '
        'Only one source encoder was tested, with three fitted readouts.')
    md('runtime','## Encoder cost is part of the architecture choice\n\n'
        'Cached-feature training and prepared-input decoder timings exclude the full backbone. '
        'This separate audit recomputes the encoder, preprocessing and requested output on every invocation. '
        'RGB64 inputs are already on the device, so capture, file decoding, host transfer and planning are excluded. '
        'The five packages use seed9107; all ViT packages use the same original P1 pose head. '
        'Twenty synchronized repeats after five warmups give descriptive median/p95 batch latency, with batch1 and32 reported separately. '
        'Per-image throughput at batch32 is not single-frame interaction latency. Resident/peak memory covers all loaded package outputs even when one is requested.')
    table('perception_runtime')
    timings=report['datasets'].get('perception_runtime',[])
    if len(timings)==40:
        lookup={(r['package'],r['batch'],r['output']):r for r in timings}
        names=('cnn_p1','vit_p1','vit_early','vit_conditioned','vit_split')
        compact=[dict(package=name,batch1_ms=lookup[name,1,'all']['median_batch_ms'],
            batch1_p95_ms=lookup[name,1,'all']['p95_batch_ms'],
            batch32_ms_per_image=lookup[name,32,'all']['median_ms_per_image'],
            parameters=lookup[name,1,'all']['parameters']) for name in names]
        custom_table('runtime_all_outputs','Full encoder plus RGB, foreground and pose',compact,
            [('package','Package'),('batch1_ms','Single-frame median ms'),('batch1_p95_ms','Sample p95 ms'),
             ('batch32_ms_per_image','Batch32 ms/image'),('parameters','Loaded parameters')],
            'Same requested outputs and seed9107; descriptive local GPU timing. The original P1 pose head is held common across ViT decoder packages.')
        md('runtime_findings',f'The compact CNN takes {compact[0]["batch1_ms"]:.2f} ms for all three outputs on one frame; '
            f'the ViT early-input and split-trunk packages take {compact[2]["batch1_ms"]:.2f} and {compact[4]["batch1_ms"]:.2f} ms. '
            'The frozen transformer dominates these package costs, so doubling the small decoder trunk does not double full-package runtime. '
            'Batch32 throughput and single-frame latency answer different deployment questions. Desktop GPU activity and the short repeated sample limit precision; '
            'these measurements are not complete application latency or a service guarantee.')
    md('budget','## Does this establish convergence or the need for more training?\n\n'
        'The fit budget compares practical short runs; it is not a convergence proof or a scaling-law experiment. '
        'For a transparent local diagnostic, the following table compares the mean of the five recorded validation points at3100–3500 '
        'with the five at3600–4000. Ratios below1 indicate decreasing error; positive IoU changes indicate improvement. '
        'No significance threshold or asymptotic extrapolation is applied. A model can improve within its current data distribution '
        'without improving fresh geometry or downstream control. These results favor testing the missing information paths and coverage first. '
        'They do not rule out gains from longer training, a different learning rate or a larger pretrained encoder.')
    custom_table('validation_tail','Validation movement near the training budget',report['validation_tail_trends'],
        [('family','Study'),('arm','Arm'),('seed','Seed'),('q_ratio','Mean-q ratio'),
         ('coco_image_mse_ratio','COCO MSE ratio'),('mask_iou_delta','Mask IoU change')],
        'Late-window mean divided by the preceding-window mean; IoU is an absolute difference. Blank cells indicate untrained outputs, not zero. Fixed windows are descriptive and did not select checkpoints.')
    md('interface','## Recommended interface and what remains to be trained\n\n'
        'Expose a named feature collection, keeping spatial grid, channel width, originating encoder depth, source time and availability explicit. '
        'A consumer should project and fuse the levels it needs through its own adapter. Derived pooled grids must be labeled as derived; '
        'adding a pooled level does not add encoder depth. Global contextual tokens and local evidence can coexist without forcing every task '
        'through a single small projection.\n\n'
        'The practical candidate is an asymmetric architecture: pretrained contextual vision features plus a local observation path, '
        'with convolutional dense RGB/mask decoding and a spatial geometry readout. This matches the direction of '
        '[DPT](https://arxiv.org/abs/2103.13413), where transformer features are reassembled and decoded densely. '
        '[FPN](https://arxiv.org/abs/1612.03144) motivates lateral/top-down multi-scale fusion; it does not imply every level must attend to every other level. '
        '[FiLM](https://arxiv.org/abs/1709.07871) supplies a concrete conditioning mechanism whose benefit here depends on the measured tradeoff. '
        '[DINOv2](https://arxiv.org/abs/2304.07193) is the source of the tested pretrained representation, not evidence that our dynamics will work.\n\n'
        'Observation features feed the state updater together with previous memory and executed action. Candidate future actions condition the predictor. '
        'Future decoding must use predicted features and time-aligned predicted state; real future pixels or future encoder skips would leak the answer. '
        'An observation decoder may use current raw/early skips, but those inputs are unavailable for an imagined future until a causal detail model supplies them. '
        'A decoder fitted only to complete observed features must also be trained for imperfect predicted features or for an explicit missing-detail mode; '
        'simply deleting its local input at inference is not an established solution. '
        'Changing native width from64 to384 also requires new compatible state-update and predictor adapters; old dynamics checkpoints cannot be assumed transferable.\n\n'
        'Task-conditioned encoder adapters, additional memory registers, four independent spatial levels, deeper pretrained intermediate taps, '
        'queried-instance masks and arbitrary action/output modalities have not been trained in this study. They remain distinct experiments. '
        'A universal all-scales-to-all-layers network is not justified by the current evidence.')
    custom_table('routing_contract','Proposed integration contract',[
        dict(component='Observation encoder',inputs='Current observation; explicit resolution/preprocessing',
             outputs='Named local and contextual spatial features, native width, source depth/time and validity',
             status='Frozen CNN/ViT prototypes tested; general feature collection integration remains to implement'),
        dict(component='Dense output decoder',inputs='Consumer-owned projections of local + final + derived coarse evidence; optional output task',
             outputs='RGB or foreground probability at the requested observation time',status='Shared, FiLM and split-trunk prototypes compared'),
        dict(component='Geometry readout',inputs='Spatial contextual grids; object convention defined by supervision',
             outputs='Point distributions, expected positions and orientation; extent requires another output',status='Two frozen packages and objective-only follow-up tested'),
        dict(component='State updater',inputs='Previous state + current evidence + executed action + elapsed time when variable',
             outputs='Persistent state with episode reset and source-time conventions',status='New feature-width adapter and compatible training remain'),
        dict(component='Transition predictor',inputs='Current state + candidate action + horizon',
             outputs='Predicted feature/state quantities at the target time',status='Must be trained and checked for multi-step physical/action effects'),
        dict(component='Future output decoder',inputs='Predicted features and aligned predicted memory; causally available detail only',
             outputs='Future images, geometry or other requested quantities',status='Observation skips cannot supply real unseen future pixels'),
        dict(component='Task/query access',inputs='Requested output, object query or planning goal',
             outputs='Task-specific consumption of common evidence; optionally a trained encoder adapter',
             status='RGB/mask FiLM tested; queried objects and encoder conditioning remain untested')],
        [('component','Component'),('inputs','Inputs'),('outputs','Output contract'),('status','Evidence / remaining work')],
        'This is a proposed integration contract, not a claim that new state-update or prediction modules have been trained. Consumers request relevant levels; every level need not feed every layer.')
    md('encoder_conditioning','Encoder conditioning remains feasible, but its role must be explicit. '
        'Previous memory or executed actions can help interpret partial observations; a task query can request a specialized view of the evidence. '
        'For an initial integration, retain a stable observation representation and apply task context at consumer adapters. '
        'Compare earlier conditioning only with a neutral-context control and retention checks across tasks. '
        'A hypothetical future action belongs in the transition model or a planning query; do not silently redefine the observed world state for each candidate action. '
        'Conditioning does not create an untrained output capability or replace the required supervision.')
    md('data_next','## Data sufficiency and the next discriminating tests\n\n'
        'The existing COCO labels and PushT source are sufficient for the present reconstruction, foreground/category and geometry comparisons. '
        'Additional downloadable image collections would not fix the demonstrated coverage and causal-evaluation gaps by themselves. '
        'For the next geometry step, prospectively sample broader simulator configurations and trajectories, retain explicit disjoint test groups, '
        'and compare boundary/occlusion/contact coverage under a fixed budget. The current fresh cohort has already informed design and should not become the sole final test.\n\n'
        'For a world model, train a compatible state updater and action-conditioned predictor only after choosing the perception interface. '
        'Evaluate multi-step physical errors, action-shuffle/copy controls, collisions, occlusion and actual planning benefit on held-out trajectories. '
        'Computer use requires screenshot/action/transition data, enough input resolution to preserve text and controls, and task-success evaluations; '
        'upscaling the current64×64 input cannot restore lost detail. Driving additionally requires synchronized temporal, action and sensor evidence. '
        'COCO foreground masks and static PushT poses do not establish those capabilities. No albedo work is proposed.')
    md('validation','## Validation, provenance and limits\n\n'
        'The reconciliation verifies selected updates against validation ledgers, actual training counts, paired sample identities, '
        'raw pose errors, per-image RGB/mask metrics and category AP from scores with unknown labels. Reference checkpoints and all failed/stopped '
        'records are preserved. Essential checks cover feature/target alignment, frozen state, resume identity, paired initialization, '
        'attention calculations/interventions, invertible decoder input packing and boundary-exact localization targets. '
        'Each completed fit or standalone evaluation refreshes the canonical HTML and runs browser verification.\n\n'
        'Claude participated through the real installed CLI with public conceptual briefs and web-only tools. Private code, datasets and numerical results '
        'were not exported. The implementing agent verified code, calculations and actual experimental evidence locally. '
        'The collaboration corrected unsupported assumptions about patch-embedding positions, raw/early equivalence, normalization loss and task-pass parameter sharing. '
        'Agreement between reviewers is not experimental validation. Exact briefs, replies, corrections and execution receipts remain in the collaboration directory.\n\n'
        'Main limits: one source checkpoint per encoder package; architecture/pretraining/size/preprocessing confounds; three head/module seeds; reused grouped '
        'holdouts; pretrained-data overlap with COCO not independently audited; an adaptive fresh-cohort follow-up; '
        'unequal pose-selected versus fixed-endpoint rules across separate studies; decoder capacity and loss-scale choices; '
        'no new temporal or control evaluation. Publication-only directional-panel and source-inventory errors were repaired from preserved raw evaluation without repeating inference. '
        'Large source lists now retain one exact path per record key, and smaller image previews use verified lossless pixel-preserving encoding. '
        'The final full suite initially exposed one older assertion expecting the removed joined-source cell; it was migrated to verify exact run/path pairs and source counts. '
        'Both the failed receipt and corrected full-suite receipt are preserved; the corrected suite has407 passed and3 opt-in browser-transport skips, with no failures. '
        'No incomplete endpoint is silently promoted to a complete4,000-update comparison.')
    if report['pending']:
        custom_table('pending','Declared work still pending',[dict(item=p) for p in report['pending']],[('item','Pending unit')],
            'This partial report remains incomplete until all declared units finish or are explicitly recorded as stopped/failed at the deadline.')


if __name__=='__main__': build()
