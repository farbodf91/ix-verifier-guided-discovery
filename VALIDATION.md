# Validation — November 1, 2026

## Completed

- Installed the pinned requirements in a **new Python 3.12 virtual environment**, including PyTorch 2.4.1 CPU. `pip check`: no broken requirements.
- Executed the main and cylinder notebooks' enabled scientific cells in order, including setup. Optional fresh generation and optimization stayed disabled.
- Captured outputs in execution order, with each equation heading and complete system in one Markdown-math output, plus readable plain text. Notebook schema validation passed; all code cells parse; all optional anchors resolve.
- Recomputed candidate ranking examples using ODEFormer's actual integration path. No mock solver or replacement recorded predictions were used.
- Compared all full-suite default shortlist decisions with their saved records. The teaching and full-suite accepted sets agree exactly on both cases.
- Checked empty-admissible-set behavior and fallback from a rejected coefficient refinement.
- Inspected scientific figures and complete equation outputs for full terms, legible labels, aspect ratio, and overlapping annotations.
- Tested repository configuration in a separate copy: notebook URLs, Colab links, revision substitution, repeated configuration, and HTML preview regeneration.

The final revision also reran the refined-baseline comparison: median test NMSE is 2.410520 at mu=0.5 and 0.044812 at mu=1.5 for the recorded representative baseline models. The shared equation display and all enabled notebook cells completed successfully.

## Numerical results

| Case | Selected candidate | Recomputed final test NMSE |
|---|---:|---:|
| Van der Pol mu=0.5 | 118 | Median over eight ICs: 0.0006985121153988921 |
| Van der Pol mu=1.5 | 100 | Median over eight ICs: 0.0032670982694707265 |
| Cylinder rank6/decode8 | 20 | Full 150-state test interval: 0.0688674086555076 |

The Van der Pol values reproduce the prior tutorial outputs. The cylinder value also matches the previous tutorial's full-interval computation; it is distinct from window-averaged or full-field error measures in the recorded tables.

## Execution method and practical limits

This workspace does not support Jupyter's network kernel sockets. Cells were executed using a real in-process IPython shell with ordered stream/display capture, in the clean environment during dependency validation and in the same pinned environment for the final revisions. This executes the scientific code but does not emulate Colab's browser interface. `tools/execute_notebooks.py` provides standard nbclient execution for normal local environments.

A signed-in Colab run and a VS Code UI check have **not** been performed here. Saved equation outputs use Markdown math plus meaningful plain text, rather than a LaTeX-only MIME object or a raster equation image. The HTML previews were regenerated from the executed notebooks. Fresh checkpoint download and stochastic candidate generation were not rerun; the official loading path is retained and generation cells remain optional.

Notebook setup, source links, and Colab links are configured for `https://github.com/farbodf91/ix-verifier-guided-discovery` on `main`. Temporary configuration guards have been removed. Public clone and Colab execution checks remain pending file upload.
