# Scientific sources and numerical conventions

The tutorial is based on Farbod Faraji and Francesco Belardinelli (2026), [Verifier-Guided Model Discovery for Physical Dynamical Systems with Pretrained Symbolic Transformers](https://arxiv.org/abs/2608.02662). Numerical records supplied by the authors define the examples.

ODEFormer is from d'Ascoli et al. (ICLR 2024), [paper](https://arxiv.org/abs/2310.05573) and [official source](https://github.com/sdascoli/odeformer). The architecture illustration is attributed to that work. The environment uses released `odeformer==0.1.7`; upstream revision inspected during preparation was `c9193012ad07a97186290b98d8290d1a177f4609`. The official pretrained-checkpoint download mechanism is retained.

## Van der Pol data and ranking

- Twelve discovery initial conditions, sampled with NumPy seed 1 in [-1,1]^2; each trajectory has 301 samples over t=0..20.
- Each trajectory supplies early t=0..10 and late t=10..20 windows, each with 151 samples. Window times are reset to zero for integration.
- Generation uses indices `[3,4,5,11,13,16,17,18]` from the alternating early/late bank: three early and five late windows.
- All 24 windows rank the pooled candidates. Eight separate test ICs use seed 3 and are withheld from selection and refinement.
- Ranking averages time-mean squared errors divided by each window's coordinate variances plus 1e-12, then averages coordinates and windows.
- Learned equations use the native ODEFormer integration path and its timeout. Failure, nonfinite values, or incomplete trajectories return failure rather than silently substituting recorded predictions. Reference trajectories use SciPy at the original tight tolerances.

## Two-check teaching policy and full-suite reproduction

The main notebook explicitly requests `suite="teaching"`; the helpers retain `suite="full"` as their default for reproducing the submitted decisions.

1. **Derivative consistency:** compare `np.gradient(x,t,axis=0,edge_order=2)` with the equation evaluated at observed states. Normalize by derivative variance across the complete discovery-window bank. Set the threshold to twice the minimum discrepancy in the complete ten-candidate shortlist.
2. **Oscillation consistency:** require all three of the submitted amplitude, period, and cross-start consistency criteria. Candidate rollouts use three discovery initial conditions (indices 0,4,8), t=0..60, dt=0.05, with measurements at t>=30. Reference measurements use discovery observations at t>=8. Period is median peak spacing; amplitude is half the 99th–1st percentile range. Relative amplitude/period limits are 15%/10%; cross-start coefficients of variation are limited to 10%/5%.

The full suite additionally requires state norm no greater than 1.5 times the largest observed discovery norm. Full-suite flags are retained alongside teaching-suite flags, but the teaching decision does not secretly use the additional bound. Failed rollouts fail oscillation consistency.

For the two default shortlists, both policies accept exactly the same sole candidates: 118 at mu=0.5 and 100 at mu=1.5. Derivative consistency alone isolates these candidates; all ten candidates in each shortlist pass the period criterion. Amplitude agreement is another component of oscillation consistency. The optional temperature-0.2, mu=1.5 record shows candidate 101 (rank 1) rejected and candidate 100 (rank 2) selected.

## Refinement and evaluation

Constant-only Nelder–Mead minimizes mean rollout NMSE across all 24 discovery windows. Verification is repeated with the original derivative threshold. A failed refinement retains the admissible raw equation; an empty admissible shortlist returns no accepted model.

The consolidated comparison uses the refined version of the recorded representative single-trajectory baseline and the refined VG equation for each mu. The same representative test trajectory is shown for both models. The main table recomputes both models' medians across eight held-out ICs. These are results for the selected representative baseline model, not the paper's aggregate statistics across single-trajectory repetitions; those remain in the source tables.

Displayed coefficients are rounded for readability; all component equations and terms are retained. Computation uses full-precision saved expressions. Complete equations use Markdown math with meaningful plain-text output as a fallback. Expanded Van der Pol comparisons align the known and discovered terms and mark additional monomials. The symbolic library may reorder or algebraically simplify expressions during display.

## Generation settings

Optional B replots `data/vanderpol/hyperparameter_sensitivity/beam_temperature_evaluation_summary.csv` for the four submitted budget/temperature configurations. It uses the archived mu=0.5 `verified_selected` rows, their full-suite decisions, and final results after coefficient refinement. The chart is a new rendering of the recorded numerical results, not an imported original figure. It does not mix the separate earlier single-window tutorial runs into those statistics. Refined systems for all four configurations are shown in full, labeled by candidate budget and temperature.

The older `data/decoding_settings_*` tutorial runs are retained as source data but are not used in the revised lesson. No fresh model sampling is needed for the main tutorial; optional fresh generation may yield a different candidate pool.

## Cylinder flow

The fixed-Re300 artifact contains a development POD decomposition with 20 modes and standardized development/validation/test coordinate histories. Symbolic discovery uses six states. The 4/6/10/20-mode reconstruction comparison does not imply discovery of 10- or 20-dimensional ODEs.

The source uses nondimensional time t*; teaching figures label the same quantity t for readability. Development is t*=50.1..75.0, validation 75.1..85.0, and test 85.1..100.0. The solver uses elapsed time for each segment. Eight of 24 five-time-unit development windows generate candidates. The companion retains the full submitted cylinder suite: derivative agreement, percentile amplitude, dominant nonzero FFT frequency, boundedness, and cross-start consistency. The frequency descriptor differs from Van der Pol's peak-derived period.

Candidate verification spans 30 time units, measures oscillations at t>=5, and uses first/middle/last development states. The recorded full-suite amplitude/frequency limits are 15%/10%, cross-start coefficients of variation 10%/5%, and norm limit three times the maximum development norm.

The main/companion field comparison shows two recorded matched snapshots, t*=92.6 and 100.0, from `fixed_Re300_figure_payload.npz`, with the same color scale and equal spatial aspect. Complete CFD fields are not included. The companion independently integrates the accepted symbolic model and reconstructs fields from its predicted coordinates; full-field error tables remain recorded results.

## Delivery and environment

Notebooks load normal repository folders; no Base64 bundles remain. The setup installs Python 3.12-compatible pins from `requirements.txt`. CPU sessions use PyTorch's CPU wheel. Colab's official [runtime-version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html), checked November 1, lists runtime 2026.07 with Python 3.12.

The repository is [farbodf91/ix-verifier-guided-discovery](https://github.com/farbodf91/ix-verifier-guided-discovery). Notebook setup and Colab links use this destination. `main` allows Monday corrections to appear through the same links. A tested commit/tag can be pinned with `--ref`. Existing runtime checkouts are not silently overwritten; reopen in a fresh runtime after updates.

See `VALIDATION.md` for checks actually completed and remaining platform testing.
