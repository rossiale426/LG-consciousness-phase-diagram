# Analysis Code — Rossi & Smecca (2026)
## *Beyond Integration: Neural Dimensionality and the Landau-Ginzburg Physics of Awareness*

**Journal of Neuroscience** (in press)  
Preprint: bioRxiv MS ID# BIORXIV/2026/720087

---

## Authors

**Alessandro Rossi**  
University of Siena, Siena, Italy; Rehabilitative Research Foundation Gianfranco Salvini, Montevarchi, Italy  
ORCID: https://orcid.org/0000-0002-6727-4511

**Antonio Smecca**  
Istituto Nazionale di Fisica Nucleare (INFN), Roma, Italy  
ORCID: https://orcid.org/0000-0002-8887-5826

Correspondence: alessandro.rossi@unisi.it

---

## Repository structure

```
landau_ginzburg_consciousness/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
│
├── S1_simulation/
│   ├── s1_lyapunov_simulation.py      # S1.1 – Proof-of-concept simulation (Fig. S1)
│   └── s1_alpha_sensitivity.py        # S1.2 – Sensitivity of ν̂ to α (Fig. S1.2)
│
├── S2_empirical/
│   ├── s2_covariance_calibration.py   # S2.2 – Covariance matrix construction
│   ├── s2_phase_diagram.py            # S2.3 – Phase structure and thresholds (Fig. S2a)
│   ├── s2_group_statistics.py         # S2.4 – Group D ordering and stats (Fig. S2b)
│   ├── s2_susceptibility.py           # S2.5 – Susceptibility divergence (Fig. S2c)
│   ├── s2_temporal_dynamics.py        # S2.6 – Sliding-window D variance (Fig. S2d)
│   ├── s2_ledoit_wolf_pipeline.py     # S2.7 – Full LW empirical pipeline (Fig. S2e)
│   └── run_analysis.py                # S2.8 – Drop-in pipeline for real fMRI data
│
└── S3_sedation/
    └── s3_sedation_levels.py          # S3.1–S3.2 – PR across sedation levels (Fig. S3)
```

---

## Requirements

Python ≥ 3.9. Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Data

The empirical analyses target the **Luppi et al. (2019)** dataset:
- Available on request from E. A. Stamatakis (e.stamatakis@cam.ac.uk)
- Public proxy dataset: OpenNeuro **ds003171** (Kandeepan et al. 2020)
  ```bash
  pip install openneuro-py
  openneuro-py download --dataset ds003171
  ```

All scripts in `S1_simulation/` and most of `S2_empirical/` run on **synthetic data**
and are fully reproducible without access to the original dataset.

---

## Usage

### Quickstart (synthetic data, no download required)

```bash
# S1: Lyapunov simulation and critical exponent
python S1_simulation/s1_lyapunov_simulation.py

# S1: Alpha sensitivity
python S1_simulation/s1_alpha_sensitivity.py

# S2: Full synthetic pipeline
python S2_empirical/s2_ledoit_wolf_pipeline.py

# S3: Sedation levels
python S3_sedation/s3_sedation_levels.py
```

### With real fMRI data

```python
from S2_empirical.run_analysis import run_analysis
import numpy as np

# data_dict: {subject_id: np.ndarray of shape (N_ROI, T)}
data_dict = {"sub-01": np.load("sub01_fmri.npy"), ...}
results_df = run_analysis(data_dict)
print(results_df)
```

---

## Citation

If you use this code, please cite:

> Rossi A, Smecca A (2026). Beyond integration: neural dimensionality and the
> Landau-Ginzburg physics of awareness. *Journal of Neuroscience* (in press).
> bioRxiv MS ID# BIORXIV/2026/720087.

---

## License

MIT License. See LICENSE file.
