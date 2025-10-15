# Structured Scale-Free Networks

## Getting Started

Clone the repository and create a feature branch:
```bash
git clone https://github.com/ikerbenidorm/structured-scale-free-networks.git
cd structured-scale-free-networks
git checkout -b feature/your-module-name
```

After making changes, push your branch and open a Pull Request:
```bash
git add .
git commit -m "Describe your changes"
git push origin feature/your-module-name
```
Then go to GitHub and create a Pull Request from `feature/your-module-name` into `main`.

---

## Installation

It is **recommended** to use a dedicated Conda environment.

### 1. Create and activate Conda environment
```bash
conda env create --file environment.yml
conda activate complex-networks
```

### 2. (Optional) Install pip-only dependencies
```bash
pip install -r requirements.txt
```

---

## Repository Structure

- **src/**  
  Contains all Python modules. Each file implements functions for:
  - network generation  
  - analysis (degree distribution, clustering, dimension)  
  - visualization  

- **notebooks/**  
  The main Jupyter notebook (`main.ipynb`) that ties together modules in `src/` and runs end-to-end experiments.

- **data/**  
  Sample inputs, generated networks, and result files (CSV, images).

- **slides/**  
  Exported presentation files for the final group presentation.

---

## Project Overview

This group assignment for the **Complex Networks** course implements and investigates the **Structured Scale-Free Network** model:
- Generate networks at varying sizes.  
- Compute and compare the empirical degree distribution and clustering coefficient against theoretical predictions.  
- Estimate network dimension.  
- Visualize network structures under different parameter settings, with an optional animation of network assembly.

---

## Team Members

- **Victor Alexander Capa Sandoval**  
- **Diego Ismael Garcia Tripiana**  
- **Iker Lomas Javaloyes** (contact: ikerlomas@ifisc.uib-csic.es)  
- **Christian Solis Calero**  

---

## Citation

If you use this code or cite our work, please reference this repository.