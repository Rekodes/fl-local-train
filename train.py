#!/usr/bin/env python3
"""
fl-local-train — algoritmo C2D de un cliente FedAvg/FedProx.

Se ejecuta DENTRO del contenedor que lanza ocean-node, sin red, como UID 1000.
Contrato de E/S (es lo que aprueba el data owner al fijar el digest de la imagen):

ENTRADAS (solo lectura)
  /data/inputs/algoCustomData.json          hiperparámetros de la ronda (los escribe ocean-node)
  /data/inputs/<datos>.csv                  datos locales del sitio (descargados por el nodo)
  modelo global w_t, por una de dos vías:
    - `model` en algoCustomData.json: safetensors en base64 (ruta de pago, ocean-app)
    - /data/persistentStorage/<bucket>/<model>  bind-mount de Persistent Storage (laboratorio)
  `model_ref` opcional ("sha256:<hex>"): si llega, el modelo recibido tiene que coincidir

SALIDAS (esquema cerrado; nada más sale del sitio)
  /data/outputs/delta_r<RRRR>.safetensors   Δw = w_local − w_t  (float32, mismas claves que el modelo)
  /data/outputs/metrics_r<RRRR>.json        métricas agregadas (sin ejemplos ni histogramas por clase)

Variables de entorno opcionales (solo para el backend local/docker del orquestador):
  FL_INPUTS_DIR, FL_OUTPUTS_DIR, FL_PS_DIR  (por defecto /data/inputs, /data/outputs, /data/persistentStorage)
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

T_PROCESS_START = time.time()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load, save_file  # noqa: E402
from torch import nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fl_model import MODEL_VERSION, build_model, state_to_float32  # noqa: E402

ALGO_VERSION = "fl-local-train/0.4.0"

INPUTS = Path(os.environ.get("FL_INPUTS_DIR", "/data/inputs"))
OUTPUTS = Path(os.environ.get("FL_OUTPUTS_DIR", "/data/outputs"))
PS_DIR = Path(os.environ.get("FL_PS_DIR", "/data/persistentStorage"))

# Límites que el algoritmo impone aunque el orquestador pida otra cosa.
# Protegen al data owner frente a hiperparámetros que favorecen la memorización.
GUARDRAILS = {
    "local_epochs": (1, 50),
    "lr": (1e-5, 1.0),
    "batch_size": (16, 4096),
    "val_fraction": (0.05, 0.5),
    "fedprox_mu": (0.0, 10.0),
    "clip_norm": (0.0, 1e6),
    "noise_multiplier": (0.0, 100.0),
    "num_threads": (1, 16),
}
DEFAULTS = {
    "round": 0,
    "model": None,               # modelo global en base64 (safetensors); tiene prioridad sobre model_file
    "model_ref": None,           # "sha256:<hex>" esperado del modelo recibido
    "model_file": None,          # nombre del fichero del modelo global en el bucket
    "model_version": MODEL_VERSION,
    "local_epochs": 5,
    "lr": 0.05,
    "optimizer": "sgd",
    "momentum": 0.0,
    "batch_size": 64,
    "val_fraction": 0.15,
    "fedprox_mu": 0.0,
    "clip_norm": 0.0,            # 0 = sin recorte del delta
    "noise_multiplier": 0.0,     # σ; ruido N(0, (σ·C)²) sobre el delta recortado
    "seed": 1234,
    "num_threads": 1,
}


def log(msg: str) -> None:
    print(f"[{time.time() - T_PROCESS_START:7.2f}s] {msg}", flush=True)


def load_hparams() -> dict:
    path = INPUTS / "algoCustomData.json"
    raw = json.loads(path.read_text()) if path.exists() else {}
    hp = {**DEFAULTS, **{k: v for k, v in raw.items() if k in DEFAULTS}}
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        log(f"hiperparámetros ignorados (no permitidos): {unknown}")
    for k, (lo, hi) in GUARDRAILS.items():
        v = float(hp[k])
        if not (lo <= v <= hi) or math.isnan(v):
            raise ValueError(f"hiperparámetro fuera de rango: {k}={hp[k]} (permitido [{lo}, {hi}])")
    if hp["model_version"] != MODEL_VERSION:
        raise ValueError(f"model_version {hp['model_version']} ≠ {MODEL_VERSION} de esta imagen")
    if hp["optimizer"] not in ("sgd", "adam"):
        raise ValueError("optimizer debe ser 'sgd' o 'adam'")
    return hp


def find_site_csv() -> Path:
    csvs = sorted(p for p in INPUTS.iterdir() if p.suffix == ".csv" or p.name.endswith(".csv.txt"))
    if not csvs:
        # URL sin extensión: tomar cualquier fichero que no sea el JSON de hparams
        csvs = sorted(p for p in INPUTS.iterdir() if p.is_file() and p.name != "algoCustomData.json")
    if len(csvs) != 1:
        raise FileNotFoundError(f"se esperaba exactamente 1 fichero de datos en {INPUTS}, hay {[p.name for p in csvs]}")
    return csvs[0]


def find_model(model_file: str | None) -> Path:
    cands = [p for p in PS_DIR.rglob("*.safetensors")] if PS_DIR.exists() else []
    if model_file:
        cands = [p for p in cands if p.name == model_file]
    if len(cands) != 1:
        raise FileNotFoundError(f"modelo global no encontrado o ambiguo ({model_file}): {[str(p) for p in cands]}")
    return cands[0]


def load_global_model(hp: dict) -> tuple[dict[str, torch.Tensor], str, str]:
    """Devuelve el state dict, su `model_ref` y de dónde llegó."""
    if hp["model"]:
        raw, source = base64.b64decode(hp["model"], validate=True), "algocustomdata"
    else:
        path = find_model(hp["model_file"])
        raw, source = path.read_bytes(), path.name
    ref = "sha256:" + hashlib.sha256(raw).hexdigest()
    if hp["model_ref"] and hp["model_ref"] != ref:
        raise ValueError(f"el modelo recibido ({ref}) no coincide con model_ref ({hp['model_ref']})")
    return load(raw), ref, source


def load_site_data(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    x, y = arr[:, :-1], arr[:, -1].astype(np.int64)
    return x, y, digest


def split(x, y, frac: float, seed: int):
    # la partición train/val depende solo de los datos y la semilla, NO de la ronda:
    # así la validación federada es comparable entre rondas
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = max(1, int(round(frac * len(y))))
    return (x[idx[n_val:]], y[idx[n_val:]]), (x[idx[:n_val]], y[idx[:n_val]])


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def flat_norm(sd: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float((v.double() ** 2).sum()) for v in sd.values()))


def main() -> int:
    timings: dict[str, float] = {}
    t0 = time.time()
    hp = load_hparams()
    torch.set_num_threads(int(hp["num_threads"]))
    rnd = int(hp["round"])
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    data_path = find_site_csv()
    global_sd, model_ref, model_source = load_global_model(hp)
    x_np, y_np, data_digest = load_site_data(data_path)
    site_seed = int(hp["seed"]) ^ int(data_digest[:8], 16)
    (xtr, ytr), (xva, yva) = split(x_np, y_np, float(hp["val_fraction"]), site_seed)
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xva_t, yva_t = torch.from_numpy(xva), torch.from_numpy(yva)

    model = build_model(n_features=x_np.shape[1])
    model.load_state_dict(global_sd, strict=True)
    global_sd = state_to_float32(model.state_dict())
    timings["load_s"] = time.time() - t0
    log(f"ronda {rnd}: datos={data_path.name} n={len(y_np)} (train={len(ytr)}, val={len(yva)}) modelo={model_ref[:19]} ({model_source})")

    # 1) validación federada del modelo global recibido (antes de entrenar)
    t1 = time.time()
    val_loss_before, val_acc_before = evaluate(model, xva_t, yva_t)
    timings["eval_before_s"] = time.time() - t1

    # 2) entrenamiento local
    t2 = time.time()
    torch.manual_seed(site_seed + 7919 * rnd)
    if hp["optimizer"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=float(hp["lr"]), momentum=float(hp["momentum"]))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    mu = float(hp["fedprox_mu"])
    anchor = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    bs = int(hp["batch_size"])
    n = len(ytr)
    train_loss_sum, steps = 0.0, 0
    for ep in range(int(hp["local_epochs"])):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xtr_t[b]), ytr_t[b])
            if anchor is not None:
                prox = sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
                loss = loss + 0.5 * mu * prox
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(b)
            steps += 1
        train_loss_sum = ep_loss / n
        log(f"  época {ep + 1}/{hp['local_epochs']} loss={train_loss_sum:.4f}")
    timings["train_s"] = time.time() - t2

    # 3) delta, recorte y ruido opcionales
    t3 = time.time()
    local_sd = state_to_float32(model.state_dict())
    delta = {k: local_sd[k] - global_sd[k] for k in global_sd}
    raw_norm = flat_norm(delta)
    clipped = False
    C = float(hp["clip_norm"])
    if C > 0 and raw_norm > C:
        delta = {k: v * (C / raw_norm) for k, v in delta.items()}
        clipped = True
    sigma = float(hp["noise_multiplier"])
    if C > 0 and sigma > 0:
        g = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1))
        delta = {k: v + torch.randn(v.shape, generator=g) * sigma * C for k, v in delta.items()}
    for k, v in delta.items():
        if not torch.isfinite(v).all():
            raise FloatingPointError(f"delta no finito en {k}")
    val_loss_after, val_acc_after = evaluate(model, xva_t, yva_t)
    timings["post_s"] = time.time() - t3

    # 4) salidas con esquema cerrado
    tag = f"r{rnd:04d}"
    save_file(delta, str(OUTPUTS / f"delta_{tag}.safetensors"),
              metadata={"round": str(rnd), "algo_version": ALGO_VERSION, "model_version": MODEL_VERSION})
    t_end = time.time()
    timings["total_in_algo_s"] = t_end - t0
    metrics = {
        "round": rnd,
        "algo_version": ALGO_VERSION,
        "model_version": MODEL_VERSION,
        "model_ref": model_ref,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "local_steps": steps,
        "val_loss_before": val_loss_before,
        "val_acc_before": val_acc_before,
        "train_loss_last_epoch": train_loss_sum,
        "val_loss_after": val_loss_after,
        "val_acc_after": val_acc_after,
        "delta_norm_raw": raw_norm,
        "delta_clipped": clipped,
        "dp_noise_multiplier": sigma if C > 0 else 0.0,
        "timings": {k: round(v, 4) for k, v in timings.items()},
        "t_process_start": T_PROCESS_START,
        "t_algo_start": t0,
        "t_algo_end": t_end,
        "python_startup_s": round(t0 - T_PROCESS_START, 4),
    }
    (OUTPUTS / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    log(f"OK val_acc {val_acc_before:.4f} → {val_acc_after:.4f}  |Δ|={raw_norm:.4f}  train={timings['train_s']:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # el log del algoritmo es lo único que el orquestador verá del fallo
        traceback.print_exc()
        sys.exit(1)
