"""Small, explicit calculations and uncluttered figures for the IX walkthrough."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.signal import find_peaks
from IPython.display import display, Markdown
from . import workflow as w
from .figures import COLORS


def intro():
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    inputs = np.linspace(-2, 2, 21)
    outputs = inputs**2 + .08*np.random.default_rng(2).normal(size=len(inputs))
    axes[0].scatter(inputs, outputs, color=COLORS['refined'], s=25, label='Sampled observations')
    dense = np.linspace(-2.1, 2.1, 200)
    axes[0].plot(dense, dense**2, color=COLORS['data'], label=r'Example relation: $y=x^2$')
    axes[0].set(title='Functional: a relation between variables', xlabel='$x$', ylabel='$y$')
    times = np.linspace(0, 6, 25)
    observed = 2*np.exp(-.5*times) + .025*np.random.default_rng(8).normal(size=len(times))
    dense_times = np.linspace(0, 6, 300)
    axes[1].scatter(times, observed, color=COLORS['refined'], s=24, label='Sampled observations', zorder=3)
    axes[1].plot(dense_times, 2*np.exp(-.5*dense_times), color=COLORS['data'], label='Trajectory of the example law')
    axes[1].set(title=r'Dynamical: $\dot{x}=-0.5x$', xlabel='Time', ylabel='$x$')
    for axis in axes:
        axis.legend(loc='upper center', bbox_to_anchor=(.5, -.23), frameon=False, ncol=1)
    return figure


def data_roles(case):
    figure = plt.figure(figsize=(10, 4.5), layout='constrained')
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.2])
    top = figure.add_subplot(grid[0, :])
    top.axis('off')
    boxes = [(.01, 'GENERATE', '8 of the 24 windows\n3 early + 5 late', COLORS['raw']),
             (.35, 'RANK + CHECK', 'All 24 discovery windows\nCompare and assess equations', COLORS['refined']),
             (.69, 'TEST', '8 separate initial conditions\nEvaluate the final equation', COLORS['baseline'])]
    for left, title, detail, color in boxes:
        box = FancyBboxPatch((left, .1), .29, .8, boxstyle='round,pad=.01',
                             transform=top.transAxes, facecolor=color, alpha=.10, edgecolor=color)
        top.add_patch(box)
        top.text(left+.145, .72, title, ha='center', weight='bold', color=color, transform=top.transAxes)
        top.text(left+.145, .38, detail, ha='center', va='center', fontsize=10, transform=top.transAxes)
    axis = figure.add_subplot(grid[1, 0])
    trajectory = case['train'][8]
    axis.scatter(trajectory['t'][::5], trajectory['x'][::5, 0], s=10, color=COLORS['data'])
    axis.axvspan(0, 10, alpha=.12, color='#777777', label='Early: 0–10')
    axis.axvspan(10, 20, alpha=.12, color='#bbbbbb', label='Late: 10–20')
    axis.axvline(10, color='.6', lw=.8)
    axis.set(title='Two windows per discovery trajectory', xlabel='Time', ylabel='$x_0$')
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.3), ncol=2, frameon=False)
    axis = figure.add_subplot(grid[1, 1])
    discovery_states = np.array([trajectory['x0'] for trajectory in case['train']])
    test_states = np.array([trajectory['x0'] for trajectory in case['test']])
    axis.scatter(*discovery_states.T, color=COLORS['refined'], label='Discovery', s=25)
    axis.scatter(*test_states.T, color=COLORS['baseline'], marker='x', label='Test', s=35)
    axis.set(title='Test trajectories start elsewhere', xlabel='Initial $x_0$', ylabel='Initial $x_1$')
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.3), ncol=2, frameon=False)
    return figure


def derivative_agreement(times, left_side, right_side):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout='constrained')
    for coordinate, axis in enumerate(axes):
        axis.plot(times, left_side[:, coordinate], color=COLORS['data'], label='Data: numerical derivative (np.gradient)')
        axis.plot(times, right_side[:, coordinate], '--', color=COLORS['refined'], label='Right side: equation at observed states')
        axis.set(title=f'Equation {coordinate+1}', xlabel='Time', ylabel=rf'$\dot x_{{{coordinate}}}$')
    figure.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=1, frameon=False)
    return figure


def oscillation_measurement(times, prediction):
    """Show autonomous integration and peak-derived period without an FFT detour."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout='constrained')
    axes[0].plot(times, prediction[:, 0], color=COLORS['refined'])
    axes[0].scatter(times[0], prediction[0, 0], color=COLORS['data'], s=28, label='Initial condition')
    axes[0].axvspan(30, 60, color=COLORS['raw'], alpha=.10)
    axes[0].set(title='Candidate prediction from one initial condition', xlabel='Time', ylabel='$x_0$')
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.25), frameon=False)
    keep = (times >= 30) & (times <= 50)
    tail_times = times[keep]
    signal = prediction[keep, 0]
    axes[1].plot(tail_times, signal, color=COLORS['refined'])
    peaks, _ = find_peaks(signal, prominence=.1*np.ptp(signal))
    axes[1].scatter(tail_times[peaks], signal[peaks], color=COLORS['data'], s=24)
    low, high = signal.min(), signal.max()
    span = high-low
    if len(peaks) >= 2:
        first, second = peaks[:2]
        height = high+.2*span
        axes[1].annotate('', xy=(tail_times[first],height), xytext=(tail_times[second],height),
                         arrowprops={'arrowstyle':'<->', 'color':COLORS['data']})
        axes[1].text((tail_times[first]+tail_times[second])/2, height+.04*span,
                     'period', ha='center', va='bottom', fontsize=10)
        axes[1].set_ylim(low-.1*span, high+.55*span)
    axes[1].set(title='Period from peaks in the measurement interval', xlabel='Time', ylabel='$x_0$')
    return figure


def complete_workflow(mu):
    """Apply the same recorded pool, live checks, and refinement gate to either case."""
    case = w.make_case(mu)
    models = w.saved_models(mu)
    shortlist = w.load_pool(mu).iloc[:10].copy()
    checked = w.verify_shortlist(shortlist, case)
    selected = w.select_admissible(checked)
    if selected is None:
        raise RuntimeError(f'No admissible equation for mu={mu}.')
    final_equation, final_checks, decision = w.accept_refinement(
        selected.equation, models['vg_refined'], case, selected.local_dynamics_limit)
    return {'case': case, 'models': models, 'checked': checked, 'selected': selected,
            'equation': final_equation, 'checks': final_checks, 'decision': decision}


def combined_results(results):
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.7), layout='constrained')
    for row, result in enumerate(results):
        case = result['case']
        models = result['models']
        test = case['test'][models['test_index']]
        baseline = w.rollout(models['baseline_refined'], test['t'], test['x0'])
        final = w.rollout(result['equation'], test['t'], test['x0'])
        curves = [('ODEFormer baseline, refined', baseline, COLORS['baseline'], '--'),
                  ('VG, refined', final, COLORS['refined'], '-')]
        for coordinate in range(2):
            axis = axes[row, coordinate]
            axis.scatter(test['t'][::6], test['x'][::6, coordinate], s=12, color=COLORS['data'], label='Test observations', zorder=5)
            for label, prediction, color, line in curves:
                if prediction is not None:
                    axis.plot(test['t'], prediction[:, coordinate], line, color=color, label=label, lw=1.5)
            axis.set(xlabel='Time from initialization', ylabel=rf'$x_{{{coordinate}}}$', title=rf'$\mu={case["mu"]:g}$ · coordinate {coordinate+1}')
        axis = axes[row, 2]
        axis.plot(*test['x'].T, color=COLORS['data'], lw=1.3)
        for label, prediction, color, line in curves:
            if prediction is not None:
                axis.plot(*prediction.T, line, color=color, lw=1.3)
        axis.set(title=rf'$\mu={case["mu"]:g}$ · phase portrait', xlabel='$x_0$', ylabel='$x_1$')
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
    return figure


def result_table(results):
    rows = []
    for result in results:
        case = result['case']
        baseline_errors = w.score_bank(result['models']['baseline_refined'], case['test'])
        final_errors = w.score_bank(result['equation'], case['test'])
        rows.append({'mu':case['mu'], 'Refined baseline median NMSE':np.median(baseline_errors),
                     'Final VG median NMSE':np.median(final_errors), 'Test initial conditions':len(case['test'])})
    return pd.DataFrame(rows)


def cylinder_bridge():
    """Two matched recorded times; preserve the spatial aspect ratio."""
    with np.load(w.CYLINDER/'reference_results/details/fixed_Re300_figure_payload.npz') as payload:
        times = payload['snapshot_time_star']
        fields = [payload[name] for name in ('truth_images','pod_images','vg_images')]
        extent = payload['extent_star']
        limit = float(payload['field_limit'])
    artifact = np.load(w.CYLINDER/'fixed_re300_pod_decomposition_rank20.npz')
    energy = np.cumsum(artifact['singular_values']**2)/artifact['total_training_fluctuation_energy']
    figure = plt.figure(figsize=(11, 8.1), layout='constrained')
    grid = figure.add_gridspec(3, 3, height_ratios=[1, 1, 1])
    axis = figure.add_subplot(grid[0, 0])
    axis.plot(np.arange(1, 21), 100*energy, 'o-', ms=3, color=COLORS['data'])
    axis.axvline(6, color=COLORS['refined'], ls='--')
    axis.set(xlabel='Retained POD modes', ylabel='Energy (%)', title='Reconstruction fidelity', xticks=[1,6,10,20])
    note = figure.add_subplot(grid[0, 1:]); note.axis('off')
    note.text(.03, .7, 'Six coordinates → six evolution equations', fontsize=13, weight='bold')
    note.text(.03, .35, 'More POD modes can improve reconstruction.\nThe VG backbone supports at most six states.', fontsize=11)
    field_axes = []
    for row, snapshot in enumerate([1, 2], start=1):
        for column, (field, title) in enumerate(zip(fields, ['Simulation','Six-mode POD','Symbolic prediction'])):
            axis = figure.add_subplot(grid[row, column])
            image = axis.imshow(field[snapshot], origin='lower', extent=extent, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='equal')
            axis.add_patch(Circle((0,0), .5, facecolor='white', edgecolor='black', lw=.8))
            axis.set(title=f'{title} · t = {times[snapshot]:.1f}', xlabel='$x/D$', ylabel='$y/D$')
            field_axes.append(axis)
    figure.colorbar(image, ax=field_axes, shrink=.75, label='Nondimensional vorticity')
    return figure


def cylinder_equation():
    model = json.loads((w.ROOT/'data/cylinder_model.json').read_text())
    return model['optimized_equation']


def beam_summary(records, mu=0.5):
    """Final refined VG results from the submitted sensitivity experiment."""
    subset = records.loc[(records.mu == mu) & (records.selection_branch == 'verified_selected')]
    figure, axis = plt.subplots(figsize=(6.8, 3.6), layout='constrained')
    labels = [f"{row.beam_size} / {row.temperature:g}" for row in subset.itertuples()]
    positions = np.arange(len(subset))
    axis.semilogy(positions, subset.optimised_heldout_median, 'o-', color=COLORS['refined'])
    axis.set(xticks=positions, xticklabels=labels, xlabel='Candidate budget / temperature',
             ylabel='Median test NMSE', title=rf'$\mu={mu:g}$ · final refined VG')
    return figure
