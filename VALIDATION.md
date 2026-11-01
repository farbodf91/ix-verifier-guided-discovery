# Numerical checks and execution environment

The tutorial was checked with the pinned dependencies in a fresh Python 3.12 environment, using PyTorch 2.4.1 CPU. `pip check` reported no broken requirements.

## Completed checks

- Executed the enabled main and cylinder notebook cells in order, including setup. Optional fresh generation and optimization remained disabled.
- Validated notebook structure, code syntax, and internal section links.
- Recomputed ranking examples and predictions with ODEFormer's timeout-protected integration path.
- Compared default shortlist decisions with the recorded experiments. The teaching and full verifier suites selected the same candidates in both Van der Pol cases.
- Checked that an empty admissible shortlist returns no model and that a rejected coefficient refinement retains the admissible original equation.
- Inspected figures and complete equation displays for legibility.

`tools/check_recorded_results.py` reproduces the selection, error, and fallback checks for Van der Pol and checks that the cylinder prediction completes.

## Numerical results

| Case | Selected candidate | Recomputed final test NMSE |
|---|---:|---:|
| Van der Pol μ=0.5 | 118 | Median over eight ICs: 0.0006985121153988921 |
| Van der Pol μ=1.5 | 100 | Median over eight ICs: 0.0032670982694707265 |
| Cylinder rank6/decode8 | 20 | Full 150-state test interval: 0.0688674086555076 |

For the recorded representative baseline models after refinement, median test NMSE is 2.410520 at μ=0.5 and 0.044812 at μ=1.5. These values describe the selected models evaluated across eight test initial conditions, rather than an aggregate across baseline model repetitions.

The cylinder value measures the full test interval in reduced coordinates. Window-averaged and reconstructed-field errors in the source tables are different measures.

## Execution coverage

Scientific cells were executed through an in-process IPython shell. This checks Python execution and captures outputs, but does not test the Colab or VS Code interfaces. `tools/execute_notebooks.py` provides standard Jupyter execution and HTML preview generation.

Colab and VS Code interface testing remains pending. Fresh pretrained-checkpoint download and stochastic candidate generation were not rerun; those operations remain optional. Saved equations include Markdown math and readable plain text, and the HTML previews include executed outputs.
