#!/usr/bin/env python3
# kmed_infant_run.py
# ------------------------------------------------------------------------------
# KMED-I (Infancy) — Cry–Response Dyad Simulator
# Computational modelling of newborn crying as epistemic event and caregiver
# responses as policies (fiduciary, inconsistent, neglect, silencing).
# ------------------------------------------------------------------------------
# Author:          Peter Kahl
# First published: London, 30 September 2025
# Version:         0.9.7 (2025-09-30)
# License:         MIT (see below)
# Repository:      https://github.com/Peter-Kahl/KMED-Infancy
#
# © 2025 Peter Kahl / Lex et Ratio Ltd.
# ------------------------------------------------------------------------------
#
# Usage examples:
#   python kmed_infant_run.py --policy fiduciary --T 120
#   python kmed_infant_run.py --policy inconsistent --T 120 --seed 7
#   python kmed_infant_run.py --policy neglect --T 120 --noise 0.02
#   python kmed_infant_run.py --policy silencing --T 120 --save_raw
#   python kmed_infant_run.py --policy sweep --T 120 --sweep_grid 21 --save_raw
#
# Arguments:
#   --policy      fiduciary | inconsistent | neglect | silencing | sweep (default: fiduciary)
#   --T           number of cry–response cycles (default: 100)
#   --seed        RNG seed (default: 42)
#   --noise       Gaussian noise std for state updates (default: 0.01)
#   --save_raw    save raw arrays to outputs/ (default: False)
#
# Advanced (policy parameters; paper §5.3):
#   --alpha       EA reinforcement by recognition (default: 0.035)
#   --beta        EA erosion by suppression     (default: 0.045)
#   --gamma       DT reinforcement by recognition (default: 0.040)
#   --kappa       DT erosion by suppression       (default: 0.050)
#   --mu          D increase by suppression       (default: 0.040)
#   --nu          D decrease by recognition       (default: 0.035)
#   --eta         baseline dissonance drift       (default: 0.000)
#   --lam         dissonance relief by recognition (lambda) (default: 0.080)
#   --pi_penalty  punitive extra erosion under silencing   (default: 0.030)
#
# Sweep (for heatmaps of long-run EA×DT under R/S):
#   --sweep_grid  grid size (odd int; e.g., 21)   (default: 0 = off)
#
# Outputs:
#   - Plots (.png) saved in ../outputs/
#   - Run metadata saved as ..._runmeta.json
#   - Summary series saved as ..._series.json
#   - Optional raw arrays (.npy) if --save_raw
# ------------------------------------------------------------------------------
# MIT License (short form)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# Full license text: https://opensource.org/licenses/MIT
# ------------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import argparse, json, platform, getpass
import numpy as np
import matplotlib.pyplot as plt

__version__ = "0.9.7"

# -------- IO
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Params & State
@dataclass
class Params:
    alpha: float = 0.035   # +EA by R
    beta:  float = 0.045   # -EA by S
    gamma: float = 0.040   # +DT by R
    kappa: float = 0.050   # -DT by S
    mu:    float = 0.040   # +D  by S
    nu:    float = 0.035   # -D  by R
    eta:   float = 0.000   # +δ drift
    lam:   float = 0.080   # -δ by R
    pi_penalty: float = 0.030  # extra EA erosion under punitive silencing
    noise: float = 0.010

@dataclass
class State:
    EA: float = 0.50     # Epistemic Autonomy
    DT: float = 0.50     # Dissonance Tolerance
    D:  float = 0.48     # Dependence
    delta: float = 0.50  # dissonance intensity (δ)
    R: float = 0.0       # recognition (0/1) at step
    S: float = 0.0       # suppression (0/1) at step

# -------- Policies
def policy_probs(policy: str, t: int) -> tuple[float, float, float]:
    """Return (P(R), P(S), P(none)) for a given policy at step t."""
    if policy == "fiduciary":
        pR = 0.85; pS = 0.10
    elif policy == "inconsistent":
        base = 0.50 + 0.10 * np.sin(2*np.pi*t/9.0)  # oscillate recognition
        pR = np.clip(base + np.random.normal(0.0, 0.05), 0.0, 1.0)
        pS = np.clip(1.0 - pR - 0.10, 0.0, 1.0)
    elif policy == "neglect":
        pR = 0.15; pS = 0.75
    elif policy == "silencing":
        pR = 0.05; pS = 0.90
    else:
        raise ValueError(f"Unknown policy: {policy}")
    pNone = max(0.0, 1.0 - pR - pS)
    return pR, pS, pNone

def sample_response(pR: float, pS: float) -> tuple[int, int]:
    """Sample (R,S) as Bernoulli one-hot."""
    u = np.random.random()
    if u < pR:
        return 1, 0
    elif u < pR + pS:
        return 0, 1
    else:
        return 0, 0

# -------- Dynamics
def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def step(state: State, params: Params, policy: str, t: int) -> State:
    pR, pS, _ = policy_probs(policy, t)
    R, S = sample_response(pR, pS)

    EA = clip01(state.EA + params.alpha*R - params.beta*S
                - (params.pi_penalty*S if policy == "silencing" else 0.0)
                + np.random.normal(0.0, params.noise))
    DT = clip01(state.DT + params.gamma*R - params.kappa*S
                + np.random.normal(0.0, params.noise))
    D  = clip01(state.D  + params.mu*S    - params.nu*R
                + np.random.normal(0.0, params.noise))
    delta = clip01(state.delta + params.eta - params.lam*R
                   + np.random.normal(0.0, params.noise))

    return State(EA=EA, DT=DT, D=D, delta=delta, R=float(R), S=float(S))

# -------- Simulation
def run_sim(policy: str, T: int, seed: int, params: Params):
    np.random.seed(seed)
    s = State()
    series = {k: [] for k in ("EA","DT","D","delta","R","S")}
    for t in range(T):
        s = step(s, params, policy, t)
        for k in series:
            series[k].append(getattr(s, k))
    return series

# -------- Heatmap sweep (proxy scaling of recognition vs chosen y-axis) -----
def sweep_heatmap(T: int, seed: int, params: Params, grid: int = 21, mode: str = "suppression"):
    """
    x-axis: recognition scaling proxy (0.1..0.9)
    y-axis (mode):
      - suppression : suppression scaling proxy (0.1..0.9)
      - pi          : punitive penalty π (0.00..0.12)
      - noise       : state-update noise σ (0.00..0.05)
      - lambda      : δ-relief efficacy λ (0.02..0.16)
      - eta         : baseline δ drift η (0.00..0.08)
      - switch_time : repair switch time (fraction of T; pre=silencing, post=fiduciary)
      - init        : initial EA₀/DT₀ (0.10..0.90), both set to y
    """
    np.random.seed(seed)

    xs = np.linspace(0.1, 0.9, grid)  # recognition scaling proxy

    if mode == "suppression":
        ys = np.linspace(0.1, 0.9, grid);      y_label = "Suppression scaling (proxy)"
    elif mode == "pi":
        ys = np.linspace(0.00, 0.12, grid);    y_label = "Punitive penalty π"
    elif mode == "noise":
        ys = np.linspace(0.00, 0.05, grid);    y_label = "Noise σ"
    elif mode == "lambda":
        ys = np.linspace(0.02, 0.16, grid);    y_label = "Relief efficacy λ"
    elif mode == "eta":
        ys = np.linspace(0.00, 0.08, grid);    y_label = "Baseline δ drift η"
    elif mode == "switch_time":
        ys = np.linspace(0.05, 0.95, grid);    y_label = "Switch time (fraction of T)"
    elif mode == "init":
        ys = np.linspace(0.10, 0.90, grid);    y_label = "Initial EA₀ / DT₀"
    else:
        raise ValueError(f"Unknown sweep_y mode: {mode}")

    EA_end = np.zeros((grid, grid))
    DT_end = np.zeros((grid, grid))

    for i, rscale in enumerate(xs):
        for j, yval in enumerate(ys):
            p = Params(**asdict(params))
            # Recognition scaling always acts on alpha/gamma/nu
            p.alpha = params.alpha * rscale
            p.gamma = params.gamma * rscale
            p.nu    = params.nu    * rscale

            if mode == "suppression":
                supp_scale = yval
                p.beta  = params.beta  * supp_scale
                p.kappa = params.kappa * supp_scale
                p.mu    = params.mu    * supp_scale
                s = State()
                for t in range(T):
                    pr = min(0.95, 0.05 + rscale)
                    ps = min(0.95, 0.05 + supp_scale/1.2)
                    R = 1 if np.random.random() < pr else 0
                    S = 1 if (np.random.random() < ps and R == 0) else 0
                    EA = clip01(s.EA + p.alpha*R - p.beta*S + np.random.normal(0.0, p.noise))
                    DT = clip01(s.DT + p.gamma*R - p.kappa*S + np.random.normal(0.0, p.noise))
                    D  = clip01(s.D  + p.mu*S    - p.nu*R    + np.random.normal(0.0, p.noise))
                    delta = clip01(s.delta + p.eta - p.lam*R + np.random.normal(0.0, p.noise))
                    s = State(EA=EA, DT=DT, D=D, delta=delta, R=float(R), S=float(S))

            elif mode in ("pi","noise","lambda","eta","init","switch_time"):
                # Map y-value to parameter/initialisation
                if mode == "pi":
                    p.pi_penalty = float(yval)
                elif mode == "noise":
                    p.noise = float(yval)
                elif mode == "lambda":
                    p.lam = float(yval)
                elif mode == "eta":
                    p.eta = float(yval)

                s = State(EA=float(yval), DT=float(yval), D=0.48, delta=0.50) if mode == "init" else State()

                if mode != "switch_time":
                    # Hold suppression at a moderate proxy to isolate the y-effect
                    supp_scale = 0.45
                    p.beta  = params.beta  * supp_scale
                    p.kappa = params.kappa * supp_scale
                    p.mu    = params.mu    * supp_scale

                    for t in range(T):
                        pr = min(0.95, 0.05 + rscale)
                        ps = min(0.95, 0.05 + supp_scale/1.2)
                        R = 1 if np.random.random() < pr else 0
                        S = 1 if (np.random.random() < ps and R == 0) else 0
                        EA = clip01(s.EA + p.alpha*R - p.beta*S + np.random.normal(0.0, p.noise))
                        DT = clip01(s.DT + p.gamma*R - p.kappa*S + np.random.normal(0.0, p.noise))
                        D  = clip01(s.D  + p.mu*S    - p.nu*R    + np.random.normal(0.0, p.noise))
                        delta = clip01(s.delta + p.eta - p.lam*R + np.random.normal(0.0, p.noise))
                        s = State(EA=EA, DT=DT, D=D, delta=delta, R=float(R), S=float(S))
                else:
                    # Two-phase repair: pre silencing → post fiduciary
                    switch_at = max(1, min(T-1, int(yval * T)))
                    for t in range(T):
                        if t < switch_at:
                            pr = 0.05 + 0.20*rscale
                            ps = 0.90
                            R = 1 if np.random.random() < pr else 0
                            S = 1 if (np.random.random() < ps and R == 0) else 0
                            EA = clip01(s.EA + p.alpha*R - p.beta*S - p.pi_penalty*S + np.random.normal(0.0, p.noise))
                            DT = clip01(s.DT + p.gamma*R - p.kappa*S + np.random.normal(0.0, p.noise))
                            D  = clip01(s.D  + p.mu*S    - p.nu*R    + np.random.normal(0.0, p.noise))
                            delta = clip01(s.delta + p.eta - p.lam*R + np.random.normal(0.0, p.noise))
                        else:
                            pr = min(0.95, 0.60 + 0.40*rscale)
                            ps = 0.10
                            R = 1 if np.random.random() < pr else 0
                            S = 1 if (np.random.random() < ps and R == 0) else 0
                            EA = clip01(s.EA + p.alpha*R - p.beta*S + np.random.normal(0.0, p.noise))
                            DT = clip01(s.DT + p.gamma*R - p.kappa*S + np.random.normal(0.0, p.noise))
                            D  = clip01(s.D  + p.mu*S    - p.nu*R    + np.random.normal(0.0, p.noise))
                            delta = clip01(s.delta + p.eta - p.lam*R + np.random.normal(0.0, p.noise))
                        s = State(EA=EA, DT=DT, D=D, delta=delta, R=float(R), S=float(S))

            EA_end[j, i] = s.EA
            DT_end[j, i] = s.DT

    x_label = "Recognition scaling (proxy)"
    return xs, ys, EA_end, DT_end, x_label, y_label

# --- a semi-transparent box ---
def stamp_meta(ax, meta: dict, loc="lower right", fontsize=8):
    from matplotlib.offsetbox import AnchoredText
    lines = [
        f'{meta["version"]}  •  {meta["timestamp"]}',
        f'policy={meta["policy"]}, T={meta["T"]}, seed={meta["seed"]}',
        f'α={meta["alpha"]:.3f}, β={meta["beta"]:.3f}, γ={meta["gamma"]:.3f}, κ={meta["kappa"]:.3f}',
        f'μ={meta["mu"]:.3f}, ν={meta["nu"]:.3f}, η={meta["eta"]:.3f}, λ={meta["lam"]:.3f}',
        f'π={meta["pi_penalty"]:.3f}, noise={meta["noise"]:.3f}',
        # (optional) add a legend line specific to heatmaps:
        # 'Colour = final EA/DT (0–1)'
    ]
    at = AnchoredText("\n".join(lines), prop=dict(size=fontsize),
                      frameon=True, loc=loc, borderpad=0.6)
    at.patch.set_alpha(0.45)
    ax.add_artist(at)

def plot_series(series: dict, meta: dict, out_prefix: Path):
    t = np.arange(len(series["EA"]))
    # EA / DT / D
    plt.figure(figsize=(10, 5.6))
    ax = plt.gca()
    plt.plot(t, series["EA"], label="EA")
    plt.plot(t, series["DT"], label="DT")
    plt.plot(t, series["D"],  label="D")
    plt.xlabel("Cycle"); plt.ylabel("State")
    plt.title(f'KMED-I [{meta["policy"]}]: EA / DT / D over time')
    plt.legend(); plt.tight_layout()
    stamp_meta(ax, meta, loc="lower right", fontsize=8)
    plt.savefig(out_prefix.with_suffix(".states.png"), dpi=200); plt.close()

    # δ + responses
    plt.figure(figsize=(10, 5.6))
    ax = plt.gca()
    plt.plot(t, series["delta"], label="δ (dissonance)")
    plt.step(t, series["R"], where="post", label="R (recognition)")
    plt.step(t, series["S"], where="post", label="S (suppression)")
    plt.xlabel("Cycle"); plt.ylabel("Value")
    plt.title(f'KMED-I [{meta["policy"]}]: δ, R, S over time')
    plt.legend(); plt.tight_layout()
    stamp_meta(ax, meta, loc="lower right", fontsize=8)
    plt.savefig(out_prefix.with_suffix(".delta_RS.png"), dpi=200); plt.close()

def plot_heatmaps(xs, ys, EA_end, DT_end, out_prefix: Path, meta: dict, x_label: str, y_label: str, title_suffix: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Z, title in zip(axes, [EA_end, DT_end], ["EA (final)", "DT (final)"]):
        im = ax.imshow(Z, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto")
        ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.9)
    plt.suptitle(f"KMED-I heatmaps: final EA and DT — {title_suffix or y_label}")
    stamp_meta(axes[-1], meta, loc="lower right", fontsize=8)  # ← add the semi-transparent box
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".heatmaps.png"), dpi=200); plt.close()

# -------- Metadata
def build_meta(policy: str, T: int, seed: int, params: Params):
    return {
        "script": "kmed_infant_run.py",
        "version": __version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "python": platform.python_version(),
        "policy": policy,
        "T": int(T),
        "seed": int(seed),
        **asdict(params)
    }

# -------- Main
def main():
    ap = argparse.ArgumentParser(description="KMED-I (Infancy) cry–response simulator")
    ap.add_argument("--policy", choices=["fiduciary","inconsistent","neglect","silencing","sweep"], default="fiduciary")
    ap.add_argument("--T",     type=int,   default=100)
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--noise", type=float, default=0.01)
    # coefficients
    ap.add_argument("--alpha", type=float, default=0.035)
    ap.add_argument("--beta",  type=float, default=0.045)
    ap.add_argument("--gamma", type=float, default=0.040)
    ap.add_argument("--kappa", type=float, default=0.050)
    ap.add_argument("--mu",    type=float, default=0.040)
    ap.add_argument("--nu",    type=float, default=0.035)
    ap.add_argument("--eta",   type=float, default=0.000)
    ap.add_argument("--lam",   type=float, default=0.080)
    ap.add_argument("--pi_penalty", type=float, default=0.030)
    # sweep
    ap.add_argument("--sweep_grid", type=int, default=0)
    ap.add_argument(
        "--sweep_y",
        choices=["suppression","pi","noise","lambda","eta","switch_time","init"],
        default="suppression",
        help="Which y-axis to sweep (default: suppression scaling proxy)."
    )
    # raw
    ap.add_argument("--save_raw", action="store_true")

    args = ap.parse_args()

    params = Params(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, kappa=args.kappa,
        mu=args.mu, nu=args.nu, eta=args.eta, lam=args.lam,
        pi_penalty=args.pi_penalty, noise=args.noise
    )

    if args.policy != "sweep":
        series = run_sim(args.policy, args.T, args.seed, params)
        meta = build_meta(args.policy, args.T, args.seed, params)

        daystamp = datetime.now().strftime("%Y%m%d")
        prefix = OUTPUT_DIR / f"KMED-I_{args.policy}_{daystamp}"
        (OUTPUT_DIR / f"KMED-I_{args.policy}_{daystamp}_runmeta.json").write_text(json.dumps(meta, indent=2))
        (OUTPUT_DIR / f"KMED-I_{args.policy}_{daystamp}_series.json").write_text(json.dumps(series))

        if args.save_raw:
            for k, v in series.items():
                np.save(OUTPUT_DIR / f"KMED-I_{args.policy}_{daystamp}_{k}.npy", np.array(v))

        plot_series(series, meta, prefix)

    else:
        grid = int(args.sweep_grid) if args.sweep_grid else 21
        xs, ys, EA_end, DT_end, xlab, ylab = sweep_heatmap(args.T, args.seed, params, grid=grid, mode=args.sweep_y)
        meta = build_meta("sweep", args.T, args.seed, params)
        daystamp = datetime.now().strftime("%Y%m%d")
        prefix = OUTPUT_DIR / f"KMED-I_SWEEP_{daystamp}_{args.sweep_y}"
        (OUTPUT_DIR / f"KMED-I_SWEEP_{daystamp}_{args.sweep_y}_runmeta.json").write_text(json.dumps(meta, indent=2))
        if args.save_raw:
            np.save(OUTPUT_DIR / f"KMED-I_SWEEP_{daystamp}_{args.sweep_y}_EA_end.npy", EA_end)
            np.save(OUTPUT_DIR / f"KMED-I_SWEEP_{daystamp}_{args.sweep_y}_DT_end.npy", DT_end)
        plot_heatmaps(xs, ys, EA_end, DT_end, prefix, meta, xlab, ylab, title_suffix=args.sweep_y)

if __name__ == "__main__":
    main()