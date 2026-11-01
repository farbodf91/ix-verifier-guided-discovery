"""Teaching adaptation of the submitted Van der Pol and fixed-Re workflows.

Ranking and verification use ODEFormer's timeout-protected integration path.
Archived results are explicitly loaded, never silently substituted for live results.
"""
from functools import lru_cache
from pathlib import Path
import contextlib
import io
import json
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import regex
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.signal import find_peaks
from IPython.display import display

ROOT = Path(__file__).resolve().parents[1]
VDP = ROOT / "data" / "vanderpol"
CYLINDER = ROOT / "data" / "cylinder"
GENERATION_INDICES = np.array([3, 4, 5, 11, 13, 16, 17, 18])
BASELINE_INDICES = [4, 16, 18]
PASS_COLUMNS = ["pass_local_dynamics", "pass_boundedness", "pass_amplitude",
                "pass_period", "pass_attractor_consistency"]


def vdp_rhs(mu):
    def rhs(t, x):
        return np.array([x[1], mu * (1 - x[0]**2) * x[1] - x[0]])
    return rhs


def trajectory(rhs, x0, t):
    sol = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-12)
    if not sol.success or len(sol.t) != len(t):
        raise RuntimeError("Reference trajectory did not complete.")
    return {"t": np.asarray(t), "x": sol.y.T, "x0": np.asarray(x0)}


@lru_cache(maxsize=4)
def make_case(mu):
    t = np.linspace(0, 20, 301)
    train_ics = np.random.default_rng(1).uniform(-1, 1, (12, 2))
    test_ics = np.random.default_rng(3).uniform(-1, 1, (8, 2))
    full = [trajectory(vdp_rhs(mu), x0, t) for x0 in train_ics]
    tests = [trajectory(vdp_rhs(mu), x0, t) for x0 in test_ics]
    windows = []
    for j, item in enumerate(full):
        for k, start in enumerate([0, 150]):
            x = item["x"][start:start + 151].copy()
            windows.append({"t": t[:151].copy(), "x": x, "x0": x[0],
                            "name": f"mu{mu:g}_train_{j:02d}_w{k}",
                            "regime": "early" if k == 0 else "late"})
    return {"mu": mu, "train": full, "test": tests, "ranking_windows": windows,
            "generation_windows": [windows[i] for i in GENERATION_INDICES],
            "verification_initial_conditions": train_ics[[0, 4, 8]]}


@lru_cache(maxsize=1)
def native_integrator():
    from odeformer.model import SymbolicTransformerRegressor
    # Integration of an equation does not require transformer weights.
    return SymbolicTransformerRegressor(from_pretrained=False)


def rollout(equation, t, x0):
    """Return a complete finite native rollout, or None on failure/timeout."""
    prediction = native_integrator().integrate_prediction(
        np.asarray(t), np.asarray(x0, dtype=float), prediction=equation)
    if prediction is None:
        return None
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(t), len(x0)) or not np.isfinite(prediction).all():
        return None
    return prediction


def nmse(reference, prediction):
    if prediction is None:
        return np.inf
    return float(np.mean(np.mean((prediction-reference)**2, axis=0)
                         / (np.var(reference, axis=0) + 1e-12)))


def score_bank(equation, bank):
    errors = []
    for window in bank:
        prediction = rollout(equation, window["t"], window["x0"])
        errors.append(nmse(window["x"], prediction))
    return np.array(errors)


def expression_parts(equation):
    dim = len(equation.split("|"))
    symbols = sp.symbols(f"x_0:{dim}")
    local = {str(s): s for s in symbols}
    local.update({"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log,
                  "sqrt": sp.sqrt, "abs": sp.Abs, "tan": sp.tan, "sign": sp.sign})
    parts = [sp.sympify(e.strip(), locals=local) for e in equation.split("|")]
    return symbols, parts


def vector_field(equation, states):
    symbols, parts = expression_parts(equation)
    cols = []
    for expression in parts:
        value = np.asarray(sp.lambdify(symbols, expression, "numpy")(*states.T))
        cols.append(np.full(len(states), float(value)) if value.ndim == 0 else value)
    return np.column_stack(cols).astype(float)


def show_equation(equation, digits=4, expand=False, name="x", title=None):
    """Display full equation systems as Markdown math with readable text fallback."""
    symbols, parts = expression_parts(equation)
    replacements = {symbol: sp.Symbol(f"{name}_{j}") for j, symbol in enumerate(symbols)}
    lines, plain = [], []
    for j, part in enumerate(parts):
        expression = sp.expand(part) if expand else part
        constants = {value: sp.Float(float(value), digits) for value in expression.atoms(sp.Float)}
        expression = expression.xreplace(constants).xreplace(replacements)
        plain.append(f"d{name}_{j}/dt = {expression}")
        left = r"\dot{" + name + "}_{" + str(j) + "} &= "
        terms = expression.as_ordered_terms() if isinstance(expression, sp.Add) else [expression]
        current = left
        for index, term in enumerate(terms):
            negative = term.could_extract_minus_sign()
            sign = "- " if negative else ("+ " if index else "")
            term_text = sign + sp.latex(-term if negative else term)
            if len(current) + len(term_text) > 145 and current != left:
                lines.append(current)
                current = r"&\quad " + term_text
            else:
                current += term_text
        lines.append(current)
    math = r"\begin{aligned}" + r" \\[5pt] ".join(lines) + r"\end{aligned}"
    heading = ("**" + title + "**\n\n") if title else ""
    markdown = heading + "$$\n" + math + "\n$$"
    fallback = ((title + "\n") if title else "") + "\n".join(plain)
    display({"text/markdown": markdown, "text/plain": fallback}, raw=True)


def compare_vdp_equation(equation, mu, digits=4):
    """Expand and align ground-truth and discovered terms without dropping extras."""
    symbols, parts = expression_parts(equation)
    x0, x1 = symbols
    targets = [x1, -x0 + mu*x1 - mu*x0**2*x1]
    preferred = [[x1], [x0, x1, x0**2*x1]]
    rows, plain = [], []
    for coordinate, (target, candidate) in enumerate(zip(targets, parts)):
        polynomial = sp.Poly(sp.expand(candidate), x0, x1)
        terms = {x0**powers[0]*x1**powers[1]: coefficient for powers, coefficient in polynomial.terms()}
        ordered = preferred[coordinate] + [term for term in terms if term not in preferred[coordinate]]
        pieces = []
        for term in ordered:
            coefficient = terms.get(term, 0)
            if coefficient == 0:
                continue
            coefficient = sp.Float(float(coefficient), digits)
            body = sp.latex(abs(coefficient)*term)
            # Highlight only additional monomials; do not hide small nonzero coefficients.
            extra = term not in preferred[coordinate]
            if extra:
                body = r"\underbrace{" + body + r"}_{\mathrm{extra}}"
            sign = "- " if coefficient < 0 else ("+ " if pieces else "")
            pieces.append(sign + body)
        known_text = r"x_1" if coordinate == 0 else rf"-x_0 + {mu:g}x_1 - {mu:g}x_0^2x_1"
        rows.append(r"\text{Known:}\quad \dot{x}_{"+str(coordinate)+r"} &= "+known_text)
        rows.append(r"\text{VG:}\quad \dot{x}_{"+str(coordinate)+r"} &= "+" ".join(pieces))
        plain.append(f"Known dx_{coordinate}/dt = {target}\nVG dx_{coordinate}/dt = {sp.expand(candidate)}")
    text = f"**Expanded comparison: $\\mu={mu:g}$**\n\n$$\n" + r"\begin{aligned}" + r" \\[6pt] ".join(rows) + r"\end{aligned}" + "\n$$"
    display({"text/markdown": text, "text/plain": "\n".join(plain)}, raw=True)


def load_pool(mu):
    tag = str(mu).replace(".", "p")
    return pd.read_csv(VDP / f"selected_configuration/details/mu_{tag}/proposed_candidate_rollout_ranking.csv")


def load_shortlist(mu):
    tag = str(mu).replace(".", "p")
    return pd.read_csv(VDP / f"selected_configuration/details/mu_{tag}/proposed_compact_verifier_shortlist.csv")


def saved_models(mu):
    equations = pd.read_csv(VDP / "selected_configuration/tables/selected_equations.csv")
    cases = pd.read_csv(VDP / "selected_configuration/tables/supp_figure_vdp_representative_cases.csv")
    case = cases.loc[cases.mu == mu].iloc[0]
    baseline = equations.loc[(equations.mu == mu)
        & (equations.protocol == "single_trajectory_reference")
        & (equations.reference_repeat == case.reference_repeat)].iloc[0]
    vg = equations.loc[(equations.mu == mu)
        & (equations.selection_branch == "verified_selected")].iloc[0]
    return {"baseline_raw": baseline.raw_equation,
            "baseline_refined": baseline.optimised_equation,
            "vg_raw": vg.raw_equation, "vg_refined": vg.optimised_equation,
            "test_index": int(case.test_ic_index) if "test_ic_index" in case else int(case.test_trajectory_index),
            "baseline_window": int(baseline.reference_window_index)}


def derivative_score(equation, windows):
    """Compare both sides of the equation at observed states, not simulated states."""
    observed_derivatives = []
    for window in windows:
        derivative = np.gradient(window["x"], window["t"], axis=0, edge_order=2)
        observed_derivatives.append(derivative)
    derivative_scale = np.var(np.vstack(observed_derivatives), axis=0) + 1e-12

    window_scores = []
    for window, derivative in zip(windows, observed_derivatives):
        equation_values = vector_field(equation, window["x"])
        squared_difference = (equation_values - derivative)**2
        coordinate_errors = np.mean(squared_difference, axis=0) / derivative_scale
        window_scores.append(np.mean(coordinate_errors))
    return float(np.mean(window_scores))


def cycle_features(t, x, discard=30):
    keep = t >= discard
    y, times = x[keep, 0], t[keep]
    amplitude = .5 * (np.percentile(y, 99)-np.percentile(y, 1))
    peaks, _ = find_peaks(y, prominence=max(.05, .1*np.ptp(y)))
    period = np.median(np.diff(times[peaks])) if len(peaks) >= 3 else np.nan
    return float(amplitude), float(period)


def observed_cycle_targets(case):
    amplitudes, periods = [], []
    for tr in case["train"]:
        keep = tr["t"] >= 8
        t, y = tr["t"][keep], tr["x"][keep, 0]
        amplitudes.append(.5*(np.percentile(y, 99)-np.percentile(y, 1)))
        peaks, _ = find_peaks(y, prominence=max(.05, .1*np.ptp(y)))
        if len(peaks) >= 2:
            periods.extend(np.diff(t[peaks]))
    return np.median(amplitudes), np.median(periods)


def verify_equation(equation, case, local_limit, suite="full"):
    if suite not in ("full", "teaching"):
        raise ValueError("suite must be full or teaching")
    t = np.linspace(0, 60, 1201)
    predictions = [rollout(equation, t, x0) for x0 in case["verification_initial_conditions"]]
    local = derivative_score(equation, case["ranking_windows"])
    result = {"local_dynamics_score": local, "local_dynamics_limit": local_limit}
    finite = all(p is not None for p in predictions)
    observed_norm = np.linalg.norm(np.vstack([w["x"] for w in case["train"]]), axis=1).max()
    bounded = finite and max(np.linalg.norm(p, axis=1).max() for p in predictions) <= 1.5*observed_norm
    if finite:
        features = np.array([cycle_features(t, p) for p in predictions])
        target = np.array(observed_cycle_targets(case))
        errors = np.abs(np.median(features, axis=0)-target)/target
        cvs = np.std(features, axis=0)/(np.mean(features, axis=0)+1e-12)
    else:
        errors, cvs = np.full(2, np.inf), np.full(2, np.inf)
    result.update(bounded=bool(bounded), amplitude_error=errors[0], period_error=errors[1],
                  amplitude_cv=cvs[0], period_cv=cvs[1])
    result.update(pass_local_dynamics=bool(local <= local_limit),
                  pass_boundedness=bool(bounded), pass_amplitude=bool(errors[0] <= .15),
                  pass_period=bool(errors[1] <= .10),
                  pass_attractor_consistency=bool(cvs[0] <= .10 and cvs[1] <= .05))
    result["pass_oscillation"] = all(result[k] for k in
        ["pass_amplitude", "pass_period", "pass_attractor_consistency"])
    result["full_suite_admissible"] = all(result[k] for k in PASS_COLUMNS)
    if suite == "teaching":
        result["verifier_admissible"] = result["pass_local_dynamics"] and result["pass_oscillation"]
    else:
        result["verifier_admissible"] = result["full_suite_admissible"]
    return result


def verify_shortlist(shortlist, case, suite="full"):
    local_scores = [derivative_score(e, case["ranking_windows"]) for e in shortlist.equation]
    local_limit = 2 * np.min(local_scores)
    rows = []
    for _, row in shortlist.iterrows():
        rows.append({**row.to_dict(), **verify_equation(row.equation, case, local_limit, suite=suite)})
    return pd.DataFrame(rows)


def select_admissible(checked):
    passing = checked.loc[checked.verifier_admissible].sort_values("rollout_rank")
    return None if passing.empty else passing.iloc[0]


def accept_refinement(raw_equation, refined_equation, case, local_limit, suite="full"):
    """Retain a verified raw model if coefficient refinement fails a check."""
    raw_check = verify_equation(raw_equation, case, local_limit, suite=suite)
    if not raw_check["verifier_admissible"]:
        raise ValueError("Refinement requires an admissible starting equation.")
    check = verify_equation(refined_equation, case, local_limit, suite=suite)
    if check["verifier_admissible"]:
        return refined_equation, check, "Refined equation accepted"
    return raw_equation, raw_check, "Raw admissible equation retained"


def ranking_preview(pool, n=5):
    return pool.head(n)[["candidate_id", "rollout_rank", "rollout_score", "source_window"]]


def verifier_preview(checked):
    return checked[["candidate_id", "rollout_rank", "local_dynamics_score",
                    "period_error", "pass_local_dynamics", "pass_period",
                    "pass_boundedness", "verifier_admissible"]].copy()


def seed_decode(seed):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(beam_size=20, temperature=.1):
    """Reuse the official loader and support CPU checkpoint deserialization."""
    import torch
    from odeformer.model import SymbolicTransformerRegressor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_load = torch.load
    def load_on_device(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)
    with patch("torch.load", load_on_device):
        model = SymbolicTransformerRegressor(from_pretrained=True)
    model.model.to(device)
    model.model.device = device
    model.model.eval()
    model.set_model_args({"beam_size": beam_size, "beam_temperature": temperature})
    print(f"Frozen backbone on {device}; decoding mode: {model.model.beam_type}")
    return model


def candidate_table(model, top_k=20, source_window="live"):
    return pd.DataFrame([{"candidate_id": j, "equation": tree.infix(),
                          "source_window": source_window, "generation_rank": j}
                         for j, tree in enumerate(model.predictions[0][:top_k])],
                        columns=["candidate_id", "equation", "source_window", "generation_rank"])


def compare_decoding_settings(model, source, test, settings, seed=17):
    """Select on the input window only; evaluate that selection on a test trajectory."""
    import time
    records, candidates = [], []
    original = {k: getattr(model.model, k) for k in ("beam_size", "beam_temperature")}
    try:
        for beam_size, temperature in settings:
            model.set_model_args({"beam_size": beam_size, "beam_temperature": temperature})
            seed_decode(seed)
            start = time.perf_counter()
            model.fit(source["t"], source["x"], verbose=False, sort_candidates=False)
            table = candidate_table(model, top_k=len(model.predictions[0]), source_window=source["name"])
            table["beam_size"], table["temperature"] = beam_size, temperature
            table["input_nmse"] = [nmse(source["x"], rollout(e, source["t"], source["x0"]))
                                    for e in table.equation]
            table = table.sort_values("input_nmse", kind="stable").reset_index(drop=True)
            valid = table.loc[np.isfinite(table.input_nmse)]
            best = valid.iloc[0] if len(valid) else None
            keys = {tuple(sp.srepr(sp.expand(e)) for e in expression_parts(eq)[1]) for eq in table.equation}
            records.append({"beam_size": beam_size, "temperature": temperature,
                            "returned_candidates": len(table), "unique_equations": len(keys),
                            "finite_rollouts": len(valid),
                            "input_nmse": float(best.input_nmse) if best is not None else np.inf,
                            "heldout_nmse": nmse(test["x"], rollout(best.equation, test["t"], test["x0"])) if best is not None else np.inf,
                            "equation": best.equation if best is not None else "",
                            "elapsed_seconds": time.perf_counter()-start})
            candidates.append(table)
    finally:
        model.set_model_args(original)
    return pd.DataFrame(records), pd.concat(candidates, ignore_index=True)


def generate_pool(model, windows, seed=10, top_k=20, seeds=None):
    records, seen = [], set()
    for position, w in enumerate(windows):
        seed_decode(seed+position if seeds is None else int(seeds[position]))
        model.fit(w["t"], w["x"], verbose=False)
        for row in candidate_table(model, top_k, w["name"]).to_dict("records"):
            _, exprs = expression_parts(row["equation"])
            key = tuple(sp.srepr(sp.expand(e)) for e in exprs)
            if key not in seen:
                seen.add(key)
                row["candidate_id"] = len(records)
                records.append(row)
        print(f"Decoded window {position+1}/{len(windows)}; {len(records)} unique systems")
    return pd.DataFrame(records)


def deduplicate_candidates(candidates):
    """Keep the first instance of each algebraically expanded expression system."""
    keys = []
    for equation in candidates["equation"]:
        _, expressions = expression_parts(equation)
        keys.append(tuple(sp.srepr(sp.expand(expression)) for expression in expressions))
    unique = candidates.copy()
    unique["expression_key"] = keys
    unique = unique.drop_duplicates("expression_key").drop(columns="expression_key")
    unique = unique.reset_index(drop=True)
    unique["candidate_id"] = np.arange(len(unique))
    return unique


def rank_pool(pool, windows):
    ranked = pool.copy()
    ranked["rollout_score"] = [float(np.mean(score_bank(e, windows))) for e in ranked.equation]
    ranked = ranked.sort_values(["rollout_score", "generation_rank"]).reset_index(drop=True)
    ranked["rollout_rank"] = np.arange(1, len(ranked)+1)
    return ranked


class CoefficientOptimizer:
    """Submission's constant-only Nelder–Mead refinement."""
    pattern = (r"(?:(?<!_\d*))(?:(?<!\*\*))(?:[-+]?)?"
               r"(?:(?<=\()[-+]?)?(?:(?<=^)[-+]?)?"
               r"(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?")
    def __init__(self, equation, windows):
        self.equation, self.windows, self.history = equation, windows, []
        self.initial = np.array([float(m.group()) for m in regex.finditer(self.pattern, equation)])
    def insert(self, coefficients):
        values = iter(coefficients)
        return regex.sub(self.pattern, lambda _: f"{next(values):.16g}", self.equation)
    def objective(self, coefficients):
        error = float(np.mean(score_bank(self.insert(coefficients), self.windows)))
        if np.isfinite(error):
            self.history.append((error, np.array(coefficients).copy()))
        return error
    def run(self, maxiter=300):
        if not len(self.initial):
            return self.equation
        minimize(self.objective, self.initial, method="Nelder-Mead", options={"maxiter": maxiter})
        return self.insert(min(self.history, key=lambda v: v[0])[1]) if self.history else self.equation


def load_cylinder(rank=6):
    with np.load(CYLINDER / "fixed_re300_pod_decomposition_rank20.npz") as z:
        d = {k: z[k].copy() for k in z.files}
    result = {"rank": rank, "artifact": d, "basis": d["U_rank20"][:, :rank],
              "center": d[f"rank{rank}_coordinate_center"],
              "scale": d[f"rank{rank}_coordinate_scale"], "mean": d["mean_field"]}
    for segment in ["train", "validation", "test"]:
        x = d[f"rank{rank}_{segment}_standardized_coefficients"]
        result[segment] = {"t": np.arange(len(x))*.1, "x": x, "x0": x[0]}
    x, t = result["train"]["x"], result["train"]["t"]
    starts = np.rint(np.linspace(0, len(x)-51, 24)).astype(int)
    result["ranking_windows"] = [{"t": t[:51], "x": x[s:s+51], "x0": x[s], "name": f"train_{i:02d}"}
                         for i, s in enumerate(starts)]
    indices = np.rint(np.linspace(0, 23, 8)).astype(int)
    result["generation_windows"] = [result["ranking_windows"][i] for i in indices]
    result["generation_seeds"] = 17 + 1000*rank + indices
    config = f"rank{rank}_decode8"
    base = CYLINDER / "reference_results"
    result["pool"] = pd.read_csv(base / f"details/{config}_candidate_ranking.csv")
    result["shortlist"] = pd.read_csv(base / f"details/{config}_verifier_shortlist.csv")
    result["model"] = json.loads((base / f"models/{config}_optimized_equation.json").read_text())
    return result


def flow_features(t, x):
    amplitude = .5*(np.percentile(x, 99, axis=0)-np.percentile(x, 1, axis=0))
    power = np.abs(np.fft.rfft(x-x.mean(axis=0), axis=0))**2
    freq = np.fft.rfftfreq(len(t), d=np.median(np.diff(t)))
    dominant = freq[1:][np.argmax(power[1:], axis=0)]
    return amplitude, dominant


def verify_flow(equation, flow, local_limit):
    # Same fixed-Re settings as submission Notebook 02.
    states = flow["train"]["x"]
    t_ref = flow["train"]["t"]
    derivative = np.gradient(states, t_ref, axis=0, edge_order=2)
    local = float(np.mean(np.mean((vector_field(equation, states)-derivative)**2, axis=0)
                          / (np.var(derivative, axis=0)+1e-12)))
    target_a, target_f = flow_features(t_ref, states)
    t = np.arange(0, 30+.05, .1)
    # Match source: starts at first, middle, and last model-construction states.
    starts = np.rint(np.linspace(0, len(states)-1, 3)).astype(int)
    predictions = [rollout(equation, t, states[i]) for i in starts]
    finite = all(p is not None for p in predictions)
    bounded = finite and max(np.linalg.norm(p, axis=1).max() for p in predictions) <= 3*np.linalg.norm(states, axis=1).max()
    if finite:
        features = [flow_features(t[t>=5], p[t>=5]) for p in predictions]
        amps = np.array([f[0] for f in features]); freqs = np.array([f[1] for f in features])
        aerr = float(np.mean(abs(np.median(amps, axis=0)-target_a)/(abs(target_a)+1e-12)))
        ferr = float(np.mean(abs(np.median(freqs, axis=0)-target_f)/(abs(target_f)+1e-12)))
        acv = float(np.mean(amps.std(axis=0)/(amps.mean(axis=0)+1e-12)))
        fcv = float(np.mean(freqs.std(axis=0)/(freqs.mean(axis=0)+1e-12)))
    else:
        aerr = ferr = acv = fcv = np.inf
    flags = {"local": local <= local_limit, "bounded": bool(bounded),
             "amplitude": aerr <= .15, "frequency": ferr <= .1,
             "starting_states": acv <= .1 and fcv <= .05}
    return {"local_dynamics_score": local, "amplitude_error": aerr,
            "frequency_error": ferr, "amplitude_cv": acv, "frequency_cv": fcv,
            **flags, "verifier_admissible": all(flags.values())}


def reconstruct_field(flow, standardized):
    coefficients = np.asarray(standardized)*flow["scale"]+flow["center"]
    return flow["mean"].T + coefficients @ flow["basis"].T
