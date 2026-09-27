"""Exploratory timing of one symbolic training step and one evaluation (mixed M-config)."""
import time, torch, sys
import experiments.latent_agent as recipe
from pathwm.data import rule_world as rw
from pathwm.evaluation import rule_world as ev
from pathwm.models.latent_core import episode_loss

device = sys.argv[1]
s = recipe.SIZES["full"]
torch.manual_seed(0)
model = recipe.RuleModel(s, symbolic=True).to(device)
model.perception.requires_grad_(False)
perceive = ev.symbolic_perceiver(model.perception)
fams = ["category", "relation", "open", "close", "toggle"]
pool = tuple(r for f in fams for r in rw.family_ladder(f, 4))
train_pool = tuple(r for r, k in zip(pool, [7, 1, 1, 1] * 5) for _ in range(k))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
g = torch.Generator().manual_seed(1)

def sync():
    if device == "cuda": torch.cuda.synchronize()

def timed(fn, n):
    sync(); t = time.perf_counter()
    for _ in range(n): out = fn()
    sync(); return (time.perf_counter() - t) / n, out

n = 20
t_sample, batch = timed(lambda: rw.sample_episodes(g, train_pool, rw.KIND_SPLIT["train"], episodes=s["episodes"],
                                                     support=(4, 8, 16), queries=s["queries"]), n)
t_render, tok = timed(lambda: ev.encode_episodes(perceive, batch, device), n)
t_sym, tok2 = timed(lambda: ev.symbolic_episode_tokens(model.perception, batch, device), n)

def step():
    loss, _ = episode_loss(model.core, tok2, model.variance, key_weight=0.0, auxiliary_weight=0.0, reader="evidence")
    opt.zero_grad(); loss.backward(); opt.step()
t_step, _ = timed(step, n)
print(f"per update [{device}] sample {t_sample*1e3:.1f} ms | encode(render) {t_render*1e3:.1f} ms | "
      f"encode(symbolic) {t_sym*1e3:.1f} ms | fwd+bwd+opt {t_step*1e3:.1f} ms")
# one evaluation block as in the recipe (pool 320 eps N=128, small 320 eps N=8, heldout 320 eps)
pools = [rw.sample_episodes(torch.Generator().manual_seed(k), pool, rw.KIND_SPLIT["train"], episodes=320,
                            support=(n_s,), queries=s["queries"], p_empty=0.0) for k, n_s in ((11, 128), (17, 8))]
floors = rw.floors()
t_psample = None
sync(); t = time.perf_counter()
pools = [rw.sample_episodes(torch.Generator().manual_seed(k), pool, rw.KIND_SPLIT["train"], episodes=320,
                            support=(n_s,), queries=s["queries"], p_empty=0.0) for k, n_s in ((11, 128), (17, 8))]
t_psample = time.perf_counter() - t
sync(); t = time.perf_counter()
toks = [ev.symbolic_episode_tokens(model.perception, b, device) for b in pools]
sync(); t_ptok = time.perf_counter() - t
sync(); t = time.perf_counter()
for b, tk in zip(pools, toks):
    ev.control_metrics(model.core, perceive, b, device, floors, seed=3, reader="evidence", tokens=tk)
sync(); t_ctrl = time.perf_counter() - t
print(f"evaluation: sample 2 pools {t_psample:.2f} s (fixed, could be done once) | tokens {t_ptok:.2f} s | "
      f"control arms {t_ctrl:.2f} s (x3 pools incl. heldout in recipe)")
