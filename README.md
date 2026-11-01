# Symbolic Transformers and Executable Verification for Dynamical Modeling

IX/Digital Foundry tutorial by **Farbod Faraji**, based on [Faraji and Belardinelli (2026)](https://arxiv.org/abs/2608.02662).

Learn how a symbolic transformer generates dynamical equations, how pooling and ranking compare candidates across trajectories, and how executable physical checks guide selection. The main example is Van der Pol at μ=0.5, with μ=1.5 as a comparison. A companion explores six-coordinate cylinder-flow dynamics and spatial reconstruction.

## Open in Colab

- [Main teaching notebook](https://colab.research.google.com/github/farbodf91/ix-verifier-guided-discovery/blob/main/01_IX_symbolic_discovery_tutorial.ipynb)
- [Getting started](https://colab.research.google.com/github/farbodf91/ix-verifier-guided-discovery/blob/main/00_Getting_started.ipynb)
- [Cylinder-flow companion](https://colab.research.google.com/github/farbodf91/ix-verifier-guided-discovery/blob/main/02_Cylinder_flow_exploration.ipynb)

Select **Runtime → Change runtime type → Runtime Version → 2026.07** (Python 3.12). CPU is sufficient for the main walkthrough. Run the setup cell and continue in order. Setup retrieves the repository and installs the pinned packages. Optional fresh equation generation downloads the official pretrained ODEFormer checkpoint.

## Contents

| File/folder | Purpose |
|---|---|
| `01_IX_symbolic_discovery_tutorial.ipynb` | Main teaching document and three optional explorations |
| `01_IX_symbolic_discovery_tutorial.html` | Reading preview with executed outputs |
| `00_Getting_started.ipynb` | Colab, environment, and local setup guidance |
| `02_Cylinder_flow_exploration.ipynb` | Reduced-coordinate discovery and field reconstruction |
| `tutorial_lib/` | Readable numerical calculations and figure helpers |
| `data/` | Recorded candidate equations, evaluations, and cylinder data |
| `assets/` | Attributed ODEFormer schematic |
| `requirements.txt` | Compatible package versions for Python 3.12 |
| `SOURCE_NOTES.md` | Scientific conventions and result provenance |

## Local execution

Clone the repository and use Python 3.12 in a separate environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

On Windows use `.venv\Scripts\activate`. In VS Code select this environment's kernel. Run the setup cell first.

The main notebook uses the saved candidate pool and performs numerical ranking examples, verification, and predictions live. Optional A regenerates candidates and refines coefficients. Optional B compares the submitted generation-setting experiments. Optional C shows verification changing the selected candidate.


Numerical conventions and validation are documented in `SOURCE_NOTES.md` and `VALIDATION.md`.
