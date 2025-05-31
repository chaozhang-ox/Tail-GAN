# Tail-GAN: Learning to Simulate Tail Risk Scenarios

This is the README file for the project Tail-GAN: Learning to Simulate Tail Risk Scenarios (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3812973). It provides an overview of the project structure and instructions on how to use and contribute to the codebase.

## Table of Contents

- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data](#data)
- [Computing Environment](#computing-environment)
- [Citation](#citation)


## Project Structure

```
TailGAN/
├── data/                     # raw & processed datasets
├── gen_synthetic.py      # synthetic return paths (§5.1)                  -
├── gen_static_port.py    # static portfolio matrices (§3.1)               –
├── gen_thresholds.py     # MR/TF signal thresholds (App. C.1)             –
├── Dataset.py            # PyTorch Dataset helpers                        –
├── Transform.py          # price → PnL utilities                          –
├── util.py               # differentiable sorting, misc helpers           –
├── TailGAN.py            # **Tail-GAN** training loop (Alg. 1)            Fig 3,10   Tbl 1
├── WGAN.py               # Wasserstein-GAN baseline                       –
├── GOM.py                # supervised “Generative-Only Model” (§5.3)      Fig 7      Tbl 4,5
├── Evaluation.py         # relative-error RE(N), generalisation dq, ds    Tbl 1,3,6,8
├── Rejection_rate.py     # Coverage & Score tests (§4.2)                  Tbl 2
├── EigenPort.py          # eigen-portfolio construction (§5.4)            Fig 8,14   Tbl 6,9
├── Plot_Training.py      # training-error curves                          Fig 7,9,10,14
├── Plot_Quantile_PnL.py  # rank-frequency & VaR charts                    Fig 4,11,16,17
├── Plot_Corr_Auto.py     # correlation / autocorr diagnostics             Fig 5,6,12,13 
└── README.md
```

### Script-to-paper map

| Paper artefact | Script(s) that generate it |
|----------------|---------------------------|
| **Table 1** (Main results)          | `TailGAN.py`, `Evaluation.py`|
| **Table 2** (Coverage & Score test) | `Rejection_rate.py` |
| **Table 3** (Multiple risk levels)  | `TailGAN.py`, `Evaluation.py` with `--alphas [0.01,0.05,0.10]` |
| **Table 4** (Tail-GAN vs GOM)       | `GOM.py`, `TailGAN.py`, `Evaluation.py` |
| **Table 5** (Generalisation error)  | `GOM.py` |
| **Table 6 & 9** (Rand vs Eig)       | `EigenPort.py`, `TailGAN.py`, `Evaluation.py` |
| **Table 8** (Realistic Data)        | `TailGAN.py`, `Evaluation.py` |
| **Figures 4,11,16,17** (Tail quantiles) | `Plot_Quantile_PnL.py` |
| **Figures 5,6,12,13** (Corr/ACF)    | `Plot_Corr_Auto.py` |
| **Figures 7,9,10,14** (Training curves) | `Plot_Training.py` |
| **Figure 8** (Explained variance)   | `EigenPort.py` |

*If a result is produced by several scripts (e.g. training → evaluation → plot), run them in the order listed.*


## Usage

To use the project, follow these steps:

1. Run gen_synthetic.py to create synthetic data.
2. Run gen_static_port.py and gen_thresholds.py sequentially to prepare your trading strategies.
3. Run TailGAN.py to train the model and generate new scenarios.
4. Use Evaluation.py to evaluate the model's performance on the generated scenarios.
5. Use the plotting scripts to visualize the results.


## Data
In addition to the generated synthetic data, the dataset used in the section "Application to Simulation of Intraday Market Scenarios" comes from LOBSTER (https://lobsterdata.com/), which needs to be purchased by users. For the application to simulation of intraday market scenarios, run the same procedure above.


## Computing Environment
To run the reproducibility check, the following computing environment and package(s) are required:
- Environment: These experiments were conducted on a system equipped with an Nvidia A100 GPU with 40 GB of GPU memory, an AMD EPYC 7713 64-Core Processor @ 1.80GHz with 128 cores, and 1.0TB of RAM, running Ubuntu 20.04.4 LTS. 

- Package(s): 
    - Python 3.8.18
    - PyTorch 2.0.1+cu117
    - numpy 1.22.3
    - pandas 2.0.3
    - matplotlib 3.7.1


## Citation

If you use this code, please cite the paper:

```bibtex
@article{cont2022tail,
  title   = {Tail-GAN: Learning to Simulate Tail-Risk Scenarios},
  author  = {Cont, Rama and Cucuringu, Mihai and Xu, Renyuan and Zhang, Chao},
  journal={arXiv preprint arXiv:2203.01664},
  year={2022}
  note= {Working paper, version May 2025}
}
```