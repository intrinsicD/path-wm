"""Learn, freeze, store, reconstruct and revise controlled instance details.

python -m experiments.evidence_loop --output runs/evidence_loop_s17 --seed 17
See docs/evidence-loop-plan.md for the registered scope, gates and limitations.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from pathwm.data.detail_views import sample_tiles, target_view, region_mask
from pathwm.models.detail_memory import DetailCodec, to_parts, detail_loss, decode_read
from pathwm.io import atomic_json, atomic_torch, environment, source_record, state_hash, seed_everything
from pathwm.evaluation.report import write_report
from pathwm.models.belief import BeliefAgent, BeliefCorrection, BeliefDynamics
from pathwm.models.agent import Thinker, ActionHead, ErrorMonitor
from pathwm.models.hybrid_memory import HybridMemory
from pathwm.world_state.modules import AssociationBinder, ReplaceUpdater, ContextEncoder
from pathwm.world_state.session import WorldSession
from pathwm.world_state.episodes import EpisodeClient
from pathwm.world_state.store import WorldStore


def session_modules(width, version, codec=None):
    """Small existing session shell; not claimed as a trained belief/language model."""
    names = [f"part{i}" for i in range(4)] + ["conversation_state"]
    agent = BeliefAgent(width=16, context_tokens=4, latent_groups=8, latent_codes=8,
                       evidence_tokens=4, encoders={}, decoders={} if codec is None else {"detail":codec},
                       updater=BeliefCorrection(16, 8, 8, 2), dynamics=BeliefDynamics(16, 8, 8, 2),
                       thinker=Thinker(16), memory=HybridMemory(16, recent=2, block=2, blocks=1),
                       action_head=ActionHead(16), monitor=ErrorMonitor(16)).eval()
    binder = AssociationBinder()
    binder.scorer.eval()
    contracts = {n: ("detail-code", version) for n in names}
    context = ContextEncoder(16, {n: nn.Linear(width, 16) for n in names}, contracts).eval()
    modules = dict(agent=agent, binder=binder, updater=ReplaceUpdater(width).eval(), context_encoder=context)
    return modules, contracts


def serialized_codes(codes, parts, version):
    """All evaluated instances pass through the actual portable store format.

Small isolated stores avoid conflating an unbounded benchmark dataset with one
runtime session's capacity. The session/episode integration is checked separately.
"""
    restored = []
    for i in range(len(codes)):
        store = WorldStore()
        tx = store.begin("observe", occurred_at=1., available_at=1.)
        entity = tx.create_entity()
        for k in range(4):
            evidence = tx.add_evidence("camera", "image", data=dict(part=k, pixels=parts[i,k].tolist()))
            tx.put_component(entity, f"part{k}", codes[i,k].detach(), space="detail-code",
                             model_version=version, evidence=(evidence,), role="inferred")
        store.commit(tx)
        store = WorldStore.restore(json.loads(json.dumps(store.snapshot())))
        restored.append(torch.stack([store.latest(entity, f"part{k}").tensor() for k in range(4)]))
    result = torch.stack(restored).to(codes.device)
    if not torch.equal(result, codes):
        raise RuntimeError("Portable detail store changed codes")
    return result


@torch.no_grad()
def stored_demo(model, rgb, directory):
    """The trained codec actually consumes recalled session components, not a cache."""
    version = state_hash(model)
    modules, contracts = session_modules(model.width, version, model)
    session = WorldSession(**modules)
    client = EpisodeClient(session, representations=contracts)
    client.create("instance-start", "tile", kind="instance")
    client.create("conversation-start", "dialogue")
    parts = to_parts(rgb[:1]).cpu()
    device = next(model.parameters()).device
    evidence = []
    codes = model.encode(parts.to(device), torch.ones(1, 4, dtype=torch.bool, device=device))[0].cpu()
    for i in range(4):
        e = client.observe(f"see-{i}", "tile", source="camera", modality="image",
                           occurred_at=i+1., available_at=i+1., data=dict(part=i, pixels=parts[0, i].tolist()))
        evidence.append(e)
        client.publish(f"encode-{i}", "tile", f"part{i}", codes[i], evidence=(e,))
    read = client.load("tile", [f"part{i}" for i in range(4)])
    def render(c, r):
        return decode_read(model, c, r, 1, version=version)[0]
    before = render(client, read)
    direct = model.decode(codes[None].to(device), torch.ones(1,4,dtype=torch.bool,device=device), torch.tensor([1],device=device))[0]
    stored_direct_exact = torch.equal(before, direct)
    session.save(directory / "session.pt")
    restored = WorldSession.restore(torch.load(directory / "session.pt", weights_only=True), **modules)
    other = EpisodeClient(restored, representations=contracts)
    restored_read = other.load("tile", [f"part{i}" for i in range(4)])
    restart_exact = torch.equal(before, render(other, restored_read))
    # A source-issued replacement removes its old code; missing state is explicit.
    replacement = 1 - parts[:, :1]
    e = other.observe("correction-view", "tile", source="camera", modality="image",
                      occurred_at=5., available_at=5., supersedes=evidence[0],
                      data=dict(part=0, pixels=replacement[0, 0].tolist()))
    try:
        other.validate(restored_read)
        stale_rejected = False
    except ValueError:
        stale_rejected = True
    missing = other.load("tile", [f"part{i}" for i in range(4)])
    unavailable = "part0" in missing.omitted
    revised_parts = parts.clone(); revised_parts[:, 0] = replacement[:, 0]
    fresh = model.encode(revised_parts.to(device), torch.ones(1, 4, dtype=torch.bool, device=device))[0].cpu()
    other.publish("reencode-0", "tile", "part0", fresh[0], evidence=(e,))
    changed = other.load("tile", [f"part{i}" for i in range(4)])
    old_ids = {v.name: v.id for v in restored_read.components}
    unrelated = all(v.id == old_ids[v.name] for v in changed.components if v.name != "part0")
    rebuilt = model.decode(fresh[None].to(device), torch.ones(1,4,dtype=torch.bool,device=device), torch.tensor([1],device=device))[0]
    rederived_exact = torch.equal(rebuilt, render(other, changed))
    # Same episode for text and audio, no ASR or natural language claim.
    other.utterance("u-start", "dialogue", turn_id="u1", phase="start", source="person", time=6., text="show")
    waiting = other.response("dialogue", other.load("dialogue", ["conversation_state"])) == "wait"
    other.utterance("u-cancel", "dialogue", turn_id="u1", phase="cancel", source="person", time=7.)
    cancelled = other.response("dialogue", other.load("dialogue", ["conversation_state"])) == "wait"
    other.utterance("u2-start", "dialogue", turn_id="u2", phase="start", source="person", time=8., audio_ref="source-audio-chunk")
    u = other.utterance("u2-end", "dialogue", turn_id="u2", phase="end", source="person", time=9., text="show tile")
    other.publish("dialogue-state", "dialogue", "conversation_state", fresh.mean(0), evidence=(u,),
                  parents=tuple(v.id for v in changed.components), data=dict(referent="tile", request="view", pose=1))
    working = other.load("dialogue", ["conversation_state"])
    ready = other.response("dialogue", working) == "respond"
    other.emit("reply", "dialogue", working, text="Requested view prepared", dependencies=(changed,))
    complete_waits = other.response("dialogue", working) == "wait"
    other.session.save(directory / "session-final.pt")
    outcome = dict(stored_direct_exact=stored_direct_exact, restart_exact=restart_exact, stale_rejected=stale_rejected,
                   withdrawn_detail_unavailable=unavailable, unrelated_detail_retained=unrelated,
                   rederived_exact=rederived_exact, wait_for_complete=waiting,
                   cancellation_waits=cancelled, completed_request_ready=ready,
                   response_completed_once=complete_waits,
                   weights_unchanged=version == state_hash(model))
    atomic_json(directory / "episode-checks.json", outcome)
    return outcome


@torch.no_grad()
def evaluate(model, count, seed, device, directory=None):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tiles = sample_tiles(g, count, device=device)
    rgb = tiles.repeat_interleave(4, 0)
    pose = torch.arange(4, device=device).repeat(count)
    target = target_view(rgb, pose)
    full = torch.ones(len(rgb), 4, dtype=torch.bool, device=device)
    first = full.clone(); first[:, 2:] = False
    latest = ~first
    single_codes = model.encode(to_parts(tiles), torch.ones(count,4,dtype=torch.bool,device=device))
    if directory is not None:
        single_codes = serialized_codes(single_codes, to_parts(tiles).cpu(), state_hash(model))
    codes = single_codes.repeat_interleave(4,0)
    wrong = codes.reshape(count, 4, 4, model.width).roll(1, 0).reshape_as(codes)
    def decode(z, mask, p):
        out = [model.decode(z[i:i+128], mask[i:i+128], p[i:i+128]) for i in range(0, len(z), 128)]
        return tuple(torch.cat([o[j] for o in out]) for j in (0, 1))
    before, before_lv = decode(codes, first, pose)
    after, after_lv = decode(codes, full, pose)
    only, _ = decode(codes, latest, pose)
    swapped, _ = decode(wrong, full, pose)
    ignored, _ = decode(codes, full, torch.zeros_like(pose))
    old = region_mask(first, pose); new = ~old
    mse = lambda x, mask=None: float((x-target).square().mean() if mask is None else (x-target).square()[mask].mean())
    nonidentity = pose != 0
    metrics = dict(full_mse=mse(after), no_update_mse=mse(before), latest_only_mse=mse(only),
                   wrong_instance_mse=mse(swapped), new_before_mse=mse(before,new), new_after_mse=mse(after,new),
                   old_before_mse=mse(before,old), old_after_mse=mse(after,old),
                   old_output_drift=float((after-before).square()[old].mean()),
                   unknown_variance=float(before_lv.exp()[new].mean()), known_variance=float(after_lv.exp()[new].mean()),
                   nonidentity_mse=mse(after,nonidentity), copy_mse=mse(rgb,nonidentity), pose_ignored_mse=mse(ignored,nonidentity),
                   full_90_coverage=float(((after-target).abs() <= 1.644854*torch.exp(.5*after_lv)).float().mean()),
                   unknown_90_coverage=float(((before-target).abs() <= 1.644854*torch.exp(.5*before_lv))[new].float().mean()))
    metrics["psnr"] = -10*math.log10(max(metrics["full_mse"],1e-12))
    metrics["by_pose_mse"] = [mse(after,pose==p) for p in range(4)]
    metrics["by_pose_revision_ratio"] = [mse(after,new & (pose==p)[:,None,None,None])/mse(before,new & (pose==p)[:,None,None,None]) for p in range(4)]
    metrics["by_pose_old_mse_increase"] = [mse(after,old & (pose==p)[:,None,None,None])-mse(before,old & (pose==p)[:,None,None,None]) for p in range(4)]
    covered = ((before-target).abs() <= 1.644854*torch.exp(.5*before_lv))
    per_instance = (covered & new).reshape(count,4,-1).sum((1,2)).float() / new.reshape(count,4,-1).sum((1,2))
    indices = torch.randint(count,(1000,count),generator=torch.Generator().manual_seed(240929))
    boot = per_instance.cpu()[indices].mean(1)
    metrics["unknown_90_coverage_instance_bootstrap_ci"] = torch.quantile(boot,torch.tensor([.025,.975])).tolist()
    metrics["by_pose_unknown_90_coverage"] = [float(covered[new & (pose==p)[:,None,None,None]].float().mean()) for p in range(4)]
    metrics["unknown_variance_to_empirical_mse"] = metrics["unknown_variance"]/metrics["new_before_mse"]
    metrics["known_variance_to_empirical_mse"] = metrics["known_variance"]/metrics["new_after_mse"]
    metrics["known_variance_near_floor_fraction"] = float((after_lv.exp()[new] <= 1.1e-4).float().mean())
    # Independent parts: population mean is the appropriate unknown-detail prior.
    # This finite Monte Carlo reference is estimated independently, not fitted to test.
    prior_tiles = sample_tiles(torch.Generator().manual_seed(240928), 1024, device=device)
    prior = target_view(prior_tiles.mean(0,keepdim=True).expand(len(rgb),-1,-1,-1),pose)
    metrics["population_prior_unknown_mse"] = mse(prior,new)
    metrics["lossless_geometry_reference_mse"] = 0.0  # exact target generator, not a learned baseline
    gates = dict(fidelity=max(metrics["by_pose_mse"]) <= .003,
                 revision=metrics["new_after_mse"] <= .5*metrics["new_before_mse"],
                 retention=metrics["old_after_mse"] <= metrics["old_before_mse"]+.003,
                 memory=all(metrics["full_mse"] <= .5*metrics[k] for k in ("latest_only_mse","no_update_mse","wrong_instance_mse")),
                 pose=metrics["nonidentity_mse"] <= .5*min(metrics["copy_mse"],metrics["pose_ignored_mse"]),
                 uncertainty=metrics["unknown_variance"] >= 3*metrics["known_variance"])
    if directory is not None:
        np.savez_compressed(directory / "raw.npz", target=target.cpu().numpy(), before=before.cpu().numpy(),
                            after=after.cpu().numpy(), before_logvar=before_lv.cpu().numpy(), after_logvar=after_lv.cpu().numpy(),
                            latest=only.cpu().numpy(), wrong=swapped.cpu().numpy(), pose_ignored=ignored.cpu().numpy(),
                            pose=pose.cpu().numpy(), old_region=old.cpu().numpy(), canonical=rgb.cpu().numpy())
    return metrics, gates, tiles, (target[:8], after[:8])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=17); p.add_argument("--device",default="cuda")
    p.add_argument("--steps",type=int,default=3000); p.add_argument("--batch",type=int,default=64)
    p.add_argument("--hidden",type=int,default=512); p.add_argument("--eval-count",type=int,default=512)
    p.add_argument("--stop-after",type=int); p.add_argument("--resume",type=Path)
    p.add_argument("--variant",choices=("film","linear"),default="film")
    p.add_argument("--calibration-steps",type=int,default=0)
    p.add_argument("--final-eval",action="store_true")
    args=p.parse_args()
    if args.steps<1 or args.batch<1 or args.eval_count<2: p.error("Positive steps/batch and >=2 evaluation instances required")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps: p.error("stop-after must be in 1..steps")
    if args.calibration_steps < 0: p.error("calibration-steps must be nonnegative")
    if args.final_eval and (args.eval_count != 512 or args.steps != 3000 or args.seed not in (17,29) or args.stop_after not in (None,args.steps)): p.error("Final evaluation requires registered complete run: seed17/29, 3000 steps, 512 instances")
    if args.final_eval and (args.variant != "linear" or args.batch != 64 or args.hidden != 512 or args.calibration_steps != 600): p.error("Final evaluation requires selected linear configuration, batch64, hidden512, 600 calibration steps")
    seed_everything(args.seed)
    path=args.output; path.mkdir(parents=True,exist_ok=False)
    model=DetailCodec(hidden=args.hidden,variant=args.variant).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    variance_opt=torch.optim.Adam(model.logvar.parameters(),lr=.03)
    g=torch.Generator().manual_seed(100000+args.seed)
    source=source_record(__file__,model)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(scope="Controlled RGB16 textures; supplied part identity, familiar quarter turns; no natural language/audio learning",example_labels=dict(input="Requested target view",rgb="Learned codec reconstruction; store parity checked separately"))
    atomic_json(path/"run.json",dict(identity=dict(settings=settings,environment=environment(args.device)),source=source))
    (path/"recipe.py").write_text(Path(__file__).read_text()); (path/"metrics.jsonl").write_text("")
    atomic_json(path/"status.json",dict(result="running",report="pending",step=0,error=None))
    start_step=0
    calibration_done=0
    try:
        if args.resume:
            saved=torch.load(args.resume,weights_only=True,map_location=args.device)
            for key in ("seed","steps","batch","hidden","variant","calibration_steps"):
                if saved["settings"][key]!=settings[key]: raise ValueError("Incompatible resume configuration")
            if saved["source"]["sha256"]!=source["sha256"]: raise ValueError("Resume source identity changed")
            model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"])
            variance_opt.load_state_dict(saved["variance_optimizer"])
            calibration_done=saved["calibration_step"]
            if not 0 <= calibration_done <= args.calibration_steps: raise ValueError("Invalid calibration checkpoint progress")
            g.set_state(saved["generator"].cpu()); torch.set_rng_state(saved["rng"].cpu())
            if args.device.startswith("cuda"): torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
            start_step=saved["step"]
            if not 0 <= start_step <= (args.stop_after or args.steps): raise ValueError("Resume checkpoint step exceeds requested stop")
            if calibration_done and start_step != args.steps: raise ValueError("Calibration cannot precede completed mean training")
    except Exception as e:
        atomic_json(path/"status.json",dict(result="failed",report="pending",step=0,error=str(e)))
        raise
    if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start=time.monotonic(); stop=min(args.steps,args.stop_after or args.steps)
    start_calibration=calibration_done
    def save_checkpoint(step, calibration_step):
        phase = "complete" if step==args.steps and calibration_step==args.calibration_steps else "variance_calibration" if step==args.steps else "mean_training"
        atomic_torch(path/"last.pt",dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                     variance_optimizer=variance_opt.state_dict(),calibration_step=calibration_step,phase=phase,
                     generator=g.get_state(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                     settings=settings,source=source))
    def log(row):
        with (path/"metrics.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
        print(row,flush=True)
    try:
        for step in range(start_step+1,stop+1):
            model.train()
            rgb=sample_tiles(g,args.batch,device=args.device)
            mask=(torch.rand(args.batch,4,generator=g)<torch.rand(args.batch,1,generator=g)).to(args.device)
            pose=torch.randint(4,(args.batch,),generator=g).to(args.device)
            target=target_view(rgb,pose)
            mean,lv=model(to_parts(rgb),mask,pose)
            loss=detail_loss(mean,lv,target)
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training objective")
            factor=min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in opt.param_groups: group["lr"]=.002*factor
            opt.zero_grad(set_to_none=True); loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm): raise RuntimeError("Nonfinite gradients")
            opt.step()
            if step%1000==0:
                save_checkpoint(step,calibration_done)
            if step==1 or step%100==0 or step==stop:
                log(dict(step=step,split="train",loss=float(loss.detach()),gradient_norm=float(norm)))
            if step%300==0:
                metrics,_,_,_=evaluate(model,64,240925,args.device)
                log(dict(step=step,split="validation",loss=metrics["full_mse"],**metrics))
            if time.monotonic()-start>1800: raise RuntimeError("Declared training time budget exceeded")
        nonvariance={n:v.detach().clone() for n,v in model.state_dict().items() if not n.startswith("logvar.")}
        if stop == args.steps and args.calibration_steps:
            model.eval().requires_grad_(False)
            model.logvar.requires_grad_(True)
            for calibration_done in range(calibration_done+1,args.calibration_steps+1):
                rgb=sample_tiles(g,args.batch,device=args.device)
                mask=(torch.rand(args.batch,4,generator=g)<torch.rand(args.batch,1,generator=g)).to(args.device)
                pose=torch.randint(4,(args.batch,),generator=g).to(args.device)
                mean,lv=model(to_parts(rgb),mask,pose)
                loss=.5*(lv+(target_view(rgb,pose)-mean.detach()).square()*torch.exp(-lv))
                loss=loss.mean()
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite calibration objective")
                variance_opt.zero_grad(set_to_none=True);loss.backward();variance_opt.step()
                if calibration_done%100==0:
                    save_checkpoint(stop,calibration_done)
                if calibration_done%100==0 or calibration_done==args.calibration_steps:
                    log(dict(step=stop,calibration_step=calibration_done,split="calibration",loss=float(loss.detach())))
                if time.monotonic()-start>1800: raise RuntimeError("Declared calibration time budget exceeded")
        if not all(torch.equal(v,model.state_dict()[n]) for n,v in nonvariance.items()):
            raise RuntimeError("Calibration changed reconstruction weights")
        if args.device.startswith("cuda"): torch.cuda.synchronize()
        seconds=time.monotonic()-start
        save_checkpoint(stop,calibration_done)
        model.eval().requires_grad_(False)
        metrics,gates,tiles,examples=evaluate(model,args.eval_count,240927 if args.final_eval else 240925,args.device,path)
        if args.calibration_steps:
            gates["hidden_calibration"] = .85 <= metrics["unknown_90_coverage"] <= .95 and calibration_done==args.calibration_steps
        software=stored_demo(model,tiles,path)
        atomic_torch(path/"evaluation-tiles.pt",tiles.cpu())
        atomic_json(path/"result.json",dict(metrics=metrics,gates=gates,software=software,
                    passed=all(gates.values()) and all(software.values()) and stop==args.steps,
                    calibration_step=calibration_done,calibration_preserves_mean_weights=True,
                    scope=settings["scope"],final_evaluation=args.final_eval,evaluation_seed=240927 if args.final_eval else 240925,training_complete=stop==args.steps,seconds=seconds,
                    timing_scope="This invocation training segment including periodic validation; not isolated GPU kernel time",
                    samples_per_second=max(0,stop-start_step+calibration_done-start_calibration)*args.batch/max(seconds,1e-9),
                    parameters=sum(x.numel() for x in model.parameters()),
                    metric_route="All evaluation codes serialized through WorldStore; live WorldSession parity checked separately",
                    parameter_bytes=sum(x.numel()*x.element_size() for x in model.parameters()),
                    optimizer_bytes=sum(v.numel()*v.element_size() for optimizer in (opt,variance_opt) for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None))
        log(dict(step=stop,split="test" if args.final_eval else "diagnostic_validation",loss=metrics["full_mse"],**metrics))
        atomic_json(path/"status.json",dict(result="completed",report="pending",step=stop,error=None))
        try: write_report(path,batch={"rgb":examples[0]},outputs={"rgb":examples[1]})
        except Exception as e:
            atomic_json(path/"status.json",dict(result="completed",report="failed",step=stop,error=str(e))); raise
        print("gates",gates,"software",software,flush=True)
    except Exception as e:
        state=json.loads((path/"status.json").read_text())
        if state["result"]!="completed": atomic_json(path/"status.json",dict(result="failed",report="pending",step=locals().get("step",0),error=str(e)))
        raise


if __name__=="__main__": main()
