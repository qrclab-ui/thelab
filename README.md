# QRC-Lab

**QRC-Lab** is a modular, reproducible, and educational toolbox for **Quantum Reservoir Computing (QRC)**, designed to support research, experimentation, and teaching in quantum machine learning.  
The framework implements fixed, randomly initialized quantum reservoirs combined with classical readout layers, enabling temporal learning on near-term quantum devices.

---
Original paper can be found here: https://arxiv.org/abs/2602.03522
---

## 🚀 Features

- **Modular QRC architecture**
  - Fixed and random quantum reservoirs
  - Clear separation between encoding, quantum dynamics, and readout
- **Multiple execution modes**
  - Ideal simulation (statevector)
  - Shot-based simulation
  - Noisy simulation and real hardware backends
- **Flexible observable extraction**
  - Local observables (`⟨Z_i⟩`)
  - Pairwise correlations (`⟨Z_i Z_j⟩`)
- **Classical readout layer**
  - Ridge Regression
  - Linear Regression
  - Logistic Regression
- **Reproducible notebooks**
  - End-to-end examples
  - Risk bound analysis
  - Hardware-aware experiments

---

## 📁 Project Structure

```text
qrc-lab/
│
├── reservoirs.py     # Quantum reservoir definitions
├── simulator.py      # QRC execution and backend orchestration
├── observables.py    # Observable estimation and feature extraction
├── readout.py        # Classical readout models
│
├── notebooks/
│   ├── 01_intro.ipynb
│   ├── 02_risk_bounds.ipynb
│   ├── 03_real_hardware.ipynb
│   └── 04_risk_bounds_hardware.ipynb
│
└── README.md
