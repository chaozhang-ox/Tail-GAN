# Tail-GAN: Learning to Simulate Tail Risk Scenarios

This is the README file for the project Tail-GAN: Learning to Simulate Tail Risk Scenarios (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3812973). It provides an overview of the project structure and instructions on how to use and contribute to the codebase.

## Table of Contents

- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data](#data)
- [Computing Environment](#computing-environment)

## Project Structure

The project is organized as follows:

- `Dataset.py`: Characterizes a dataset
- `util.py`: Neural Sorting, used to facilitate the computation of VAR and ES, to stabliize the training of TailGAN.
- `gen_synthetic.py`: Generate synthetic data
- `gen_static_port.py`: Create transformation matrix used to generate static portfolios
- `gen_thresholds.py`: Estimate thresholds for generating trading signals based on training data.
- `Transform.py`: Convert the raw returns/price scenarios to strategy PnLs
- `TailGAN.py`: Implement the Tail-GAN training process.

## Usage

To use the project, follow these steps:

1. Run gen_synthetic.py to create synthetic data.
2. Run gen_static_port.py and gen_thresholds.py sequentially to prepare your trading strategies.
3. Run TailGAN.py to train the model and generate new scenarios.

## Data
In addition to the generated synthetic data, the dataset used in the section "Application to Simulation of Intraday Market Scenarios" comes from LOBSTER (https://lobsterdata.com/), which needs to be purchased by users.

## Computing Environment
To run the reproducibility check, the following computing environment and package(s) are required:
- Environment: These experiments were conducted on a system equipped with an Nvidia A100 GPU with 40 GB of GPU memory, an AMD EPYC 7713 64-Core Processor @ 1.80GHz with 128 cores, and 1.0TB of RAM, running Ubuntu 20.04.4 LTS. 

- Package(s): 
    - Python 3.8.18
    - PyTorch 2.0.1+cu117
    - numpy 1.22.3
    - pandas 2.0.3
