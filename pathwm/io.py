"""Run records, atomic checkpoints and deterministic direct-batch resume.

Checkpoints use tensors and primitive Python values, including encoded RNG state.
They can be read with torch.load(weights_only=True).
Exact replay is supported on the same device/software in FP32, direct sampling,
workers=0 and deterministic preprocessing. No claim spans different CUDA kernels.
"""

from pathlib import Path
import hashlib
from contextlib import contextmanager
import inspect
import json
import os
import platform
import random
import subprocess
import tempfile
import importlib.metadata
import numpy as np
import torch


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed, threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def source_record(recipe, model):
    root = Path(__file__).parent
    files = {str(p.resolve()): file_hash(p) for p in root.rglob("*.py")}
    recipe = Path(recipe).resolve()
    files[str(recipe)] = file_hash(recipe)
    # Custom modules beside a copied recipe are included in code identity too.
    for module in model.modules():
        try:
            p = Path(inspect.getfile(type(module))).resolve()
        except (TypeError, OSError):
            continue
        if "site-packages" not in str(p) and p.is_file():
            files[str(p)] = file_hash(p)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "files": files,
        "sha256": digest(files),
        "git_commit": commit,
        "recipe": str(recipe),
    }


def environment(device):
    versions = {}
    for name in ("torch", "numpy", "pillow", "matplotlib", "torchvision", "path-wm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "python": platform.python_version(),
        "packages": versions,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if torch.device(device).type == "cuda"
        else platform.machine(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.backends.cuda.matmul.fp32_precision
        if hasattr(torch.backends.cuda.matmul, "fp32_precision")
        else str(torch.backends.cuda.matmul.allow_tf32),
        "threads": torch.get_num_threads(),
    }


def training_mode(model):
    model.train()
    # Freeze buffers/dropout as well as parameters, including nested backbones.
    for module in model.modules():
        params = list(module.parameters())
        if params and not any(p.requires_grad for p in params):
            module.eval()


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(
            v.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return h.hexdigest()


class Run:
    """File/RNG bookkeeping only; the two training loops remain ordinary functions."""

    def __init__(
        self,
        output,
        *,
        settings,
        data,
        recipe,
        model,
        optimizer,
        device,
        resume=False,
        scheduler=None,
    ):
        self.path, self.model, self.optimizer, self.scheduler = (
            Path(output),
            model,
            optimizer,
            scheduler,
        )
        self.sampler = torch.Generator().manual_seed(settings["seed"] + 1009)
        source = source_record(recipe, model)
        self.identity = {
            "settings": settings,
            "data": data,
            "source_sha256": source["sha256"],
            "modules": repr(model),
            "trainable": [n for n, p in model.named_parameters() if p.requires_grad],
            "optimizer": repr(optimizer),
            "scheduler": None
            if scheduler is None
            else {
                "class": type(scheduler).__qualname__,
                "initial_state": scheduler.state_dict(),
            },
            "environment": environment(device),
        }
        self.identity = json.loads(json.dumps(self.identity, allow_nan=False))
        self.rows, self.step = [], 0
        if resume:
            record = json.loads((self.path / "run.json").read_text())
            if record["identity"] != self.identity:
                raise ValueError(
                    "Incompatible resume: code, settings, data, modules, optimizer or environment changed"
                )
            state = torch.load(
                self.path / "last.pt", map_location="cpu", weights_only=True
            )
            if state["schema"] != "pathwm-run-v1" or state["identity_sha256"] != digest(
                self.identity
            ):
                raise ValueError("Checkpoint identity mismatch")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(state["scheduler"])
            self.step, self.rows = state["step"], state["rows"]
            random.setstate(state["random"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if state["cuda"] is not None:
                torch.cuda.set_rng_state_all(state["cuda"])
            self.sampler.set_state(state["sampler"])
            # The atomic checkpoint defines committed progress; discard any later partial ledger.
            self._write_rows()
        else:
            self.path.mkdir(parents=True, exist_ok=False)
            atomic_json(
                self.path / "run.json",
                {
                    "schema": "pathwm-run-v1",
                    "identity": self.identity,
                    "source": source,
                    "initial_model_sha256": state_hash(model),
                },
            )
            (self.path / "recipe.py").write_text(Path(recipe).read_text())
            # Keep executable source, including uncommitted recipe/library edits.
            snapshot = {}
            for index, (name, expected) in enumerate(source["files"].items()):
                content = Path(name).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("Source changed during run initialization")
                local = f"source/{index:03d}_{Path(name).name}"
                target = self.path / local
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(content)
                snapshot[name] = local
            atomic_json(self.path / "source_index.json", snapshot)
            packages = sorted(
                f"{d.metadata['Name']}=={d.version}"
                for d in importlib.metadata.distributions()
            )
            (self.path / "environment.txt").write_text("\n".join(packages) + "\n")
            self.save()
        self.status("running", "pending")

    def status(self, result, report, error=None):
        atomic_json(
            self.path / "status.json",
            {"result": result, "report": report, "step": self.step, "error": error},
        )

    def sample(self, length, count):
        return torch.randint(length, (count,), generator=self.sampler).numpy()

    def log(self, row):
        if any(not np.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise ValueError("Nonfinite metric; refusing to record a successful update")
        self.rows.append(row)

    def _write_rows(self):
        p = self.path / "metrics.jsonl"
        text = "".join(
            json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in self.rows
        )
        temporary = p.with_suffix(".partial")
        temporary.write_text(text)
        temporary.replace(p)

    def save(self):
        state = {
            "schema": "pathwm-run-v1",
            "identity_sha256": digest(self.identity),
            "step": self.step,
            "rows": self.rows,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "random": random.getstate(),
            "numpy": tuple(
                v.tolist() if isinstance(v, np.ndarray) else v
                for v in np.random.get_state()
            ),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else None,
            "sampler": self.sampler.get_state(),
        }
        temporary = self.path / "checkpoint.partial"
        try:
            with temporary.open("wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            temporary.replace(self.path / "last.pt")
        finally:
            temporary.unlink(missing_ok=True)
        self._write_rows()


@contextmanager
def evaluation_mode(model):
    """Evaluation must not consume training RNG or mutate model buffers."""
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    )
    modes = {m: m.training for m in model.modules()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    model.eval()
    changed = []
    try:
        yield
    finally:
        for k, v in model.named_buffers():
            if not torch.equal(v, buffers[k]):
                changed.append(k)
                v.copy_(buffers[k])
        for m, mode in modes.items():
            m.training = mode
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        if changed:
            raise RuntimeError(f"Evaluation mutated model buffers: {changed}")


def resume_arguments(parser, args):
    """Restore declared CLI settings; explicit conflicting overrides fail visibly."""
    import sys

    if args.resume is None:
        return args
    if args.check:
        parser.error(
            "Use --check for a fresh recipe, or --resume to continue a checkpoint"
        )
    settings = json.loads((Path(args.resume) / "run.json").read_text())["identity"][
        "settings"
    ]
    supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
    for action in parser._actions:
        name = action.dest
        if name not in settings:
            continue
        value = settings[name]
        if supplied.intersection(action.option_strings):
            current = getattr(args, name)
            if isinstance(current, Path):
                current = str(current)
            if current != value:
                parser.error(
                    f"Cannot change --{name.replace('_', '-')} while resuming; copy the recipe for a new experiment"
                )
        else:
            setattr(
                args,
                name,
                action.type(value)
                if action.type is not None and value is not None
                else value,
            )
    return args


def load_component(module, path, name="encoder"):
    """Load plain weights, or one named component of this workbench's checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") == "pathwm-run-v1":
        prefix = name + "."
        state = {
            k.removeprefix(prefix): v
            for k, v in state["model"].items()
            if k.startswith(prefix)
        }
        if not state:
            raise ValueError(f"Checkpoint contains no component {name!r}")
    module.load_state_dict(state, strict=True)
    return module
