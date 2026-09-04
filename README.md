# SetkaFluo: a Multi-Element Detector Denoising Framework

[![DOI](https://zenodo.org/badge/1101518716.svg)](https://doi.org/10.5281/zenodo.22282669)

Tired of averaging your multi-element detector data and pretending that's the best we can do? Convinced there must be a smarter way? **SetkaFluo** is exactly that: a small library built to squeeze more information out of the data you already collect.

Using the Noise2Noise framework, SetkaFluo takes advantage of what your experiment naturally provides: repeated, independent noisy observations of the same underlying signal. Whether you're working with XRF or any modality where multiple detector elements observe the same spot, the structure is consistent and only the noise changes. That's exactly what we exploit.

A U-Net architecture handles the heavy lifting. Instead of blurring or smudging away your details, the network learns to separate actual structure from randomness, restoring that crisp, "salt-free" look without supervision, ground truth, or elaborate parameter tuning.

The library is designed to work out-of-the-box on Google Colab, including on free-tier machines (a T4 may need a coffee break or two, but it gets the job done).

To help you get started quickly, we include step-by-step Jupyter Notebooks that introduce the ideas, workflow, and practical details. The notebooks contain all essential explanations, so reading the comments is highly recommended.

## Publication

This package is the core implementation of our manuscript:

**Shishkov R, Laugros A, Vigano N, Bohic S, Karpov D ✉, Cloetens P ✉.** Self-Supervised Deep-Learning Denoising for X-ray Fluorescence Microscopy with Multi-Element Detectors. *Analytical Chemistry*. 2026; [10.1021/acs.analchem.5c05552](https://doi.org/10.1021/acs.analchem.5c05552)

---

## Quick start: Google Colab

The fastest way to explore **SetkaFluo** is to run the tutorial notebooks directly in Google Colab, using the example datasets published on Zenodo.

This setup requires **no local installation**, **no GPU configuration**, and **no manual environment management**. A free Google account is sufficient.

### 1. Download the Example Dataset (Zenodo)

The data needed for the notebooks are provided in the Zenodo archive:

**https://doi.org/10.5281/zenodo.17871605**

It contains two files:
- `input_data.zip` — fitted XRF maps and detector-element images
- `training.zip` — detector-element stacks for constructing Noise2Noise training pairs

These datasets correspond to the manuscript:

> Shishkov R, Laugros A, Vigano N, Bohic S, Karpov D, Cloetens P.  
> *Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy with Multi-Element Detectors.*  
> Analytical Chemistry (2026); [10.1021/acs.analchem.5c05552](https://doi.org/10.1021/acs.analchem.5c05552)

### 2. Set Up Your Google Drive Folder

In your Google Drive, create:
```
MyDrive/setkafluo_demo/
```

Place and unzip both archives inside this folder. After extraction, you should have:
```
MyDrive/setkafluo_demo/input_data/
MyDrive/setkafluo_demo/training/
```

### 3. Open and run the notebooks

| Notebook | Open in Colab | What it does |
|---|---|---|
| `01_data_exploration.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redrodion/setkafluo/blob/main/01_data_exploration.ipynb) | Load fitted XRF maps, inspect detector-element images and spectra, and explore basic visualisation options. |
| `02_denoising_prep.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redrodion/setkafluo/blob/main/02_denoising_prep.ipynb) | Build Noise2Noise training pairs by splitting detector-element maps into two independent groups, extract patch datasets, and prepare input/target tensors. |
| `03_denoising_params.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redrodion/setkafluo/blob/main/03_denoising_params.ipynb) | Experiment with training hyperparameters (patch size, batch size, learning rate, detector-element grouping) and inspect their effect on convergence and metrics. |
| `04_denoising_main.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redrodion/setkafluo/blob/main/04_denoising_main.ipynb) | Run the main training workflow for the U-Net denoiser on Siemens star and/or cell datasets, saving trained models and logs. |
| `05_denoising_compare.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redrodion/setkafluo/blob/main/05_denoising_compare.ipynb) | Apply trained models to low-dose maps, compare against classical denoisers and high-dose references, and reproduce the quantitative metrics and figures reported in the paper. |

In each notebook, the first code cell installs SetkaFluo from GitHub; an early cell mounts your Google Drive. All paths expect the dataset under `MyDrive/setkafluo_demo/`. Run the notebooks in order, `01` → `05`. A free T4 runtime suffices.

Prefer your own machine? See [Local installation](#local-installation).

---

## Local installation

We strongly recommend using a **dedicated virtual environment** for this project.

TensorFlow is the core dependency and should be installed following the official
instructions for your OS and hardware (CPU/GPU). For this reason, **TensorFlow is
not a declared dependency of the package** – you install it first, then install
the package. (If you prefer, `pip install -e ".[tf]"` installs it as an extra.)

### 1. Create a virtual environment and install TensorFlow

Follow the official TensorFlow guide for creating a virtual environment and
installing TensorFlow with `pip`:

- Official installation guide (pip + venv):  
  https://www.tensorflow.org/install/pip

In short (Linux/macOS example):
```bash
# Create and activate a virtual environment (name it as you like)
python3 -m venv tf-env
source tf-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install TensorFlow (CPU only)
pip install tensorflow

# or, for GPU support (see TF docs for details)
# pip install "tensorflow[and-cuda]"
```

Make sure you can import TensorFlow inside the environment:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

> **Note:** If you prefer conda, you can still use `conda` to manage Python
> but install TensorFlow itself with `pip` inside the environment, as recommended
> in the official docs.

### 2. Clone and install the package

With the virtual environment **activated**:
```bash
git clone https://github.com/redrodion/setkafluo.git
cd setkafluo
pip install -e .
```

This installs all dependencies **except TensorFlow**, which you already installed
in step 1. Alternatively, install TensorFlow in one shot as an optional extra:
```bash
pip install -e ".[tf]"
```

---

## How to use

Once the environment is set up and dependencies are installed:

1. **Use the library in your own scripts**

   Import the core functions directly:
```python
   from setkafluo.denoise import (
       make_unet,
       make_dataset,
       train,
       predict_tiled,
   )

   from setkafluo.data_explorer import (
       load_npz_cube_channels_last,
       sum_channels_window,
   )
```

   See the notebooks for concrete examples of how to construct training pairs,
   configure the model, and run inference on large XRF maps.

2. **Run the tests**

   With the dev extra installed (`pip install -e ".[dev]"`):
```bash
   pytest
```

3. **Notebooks on your own machine**

   The notebooks are written for Colab: they mount Google Drive and read from
   `/content/drive/MyDrive/setkafluo_demo/`. To run them locally, skip the install
   and Drive-mount cells and set the base-path variable to a local copy of the
   `setkafluo_demo/` tree.

---

## Repository overview

The repository is organized in two main parts:

- **`setkafluo/` – core library**

  This folder contains the reusable Python code that implements the SetkaFluo pipeline.
  It can be imported from your own scripts or used directly in the notebooks.

  - `setkafluo/data_explorer.py`  
    Utilities for loading and exploring XRF hyperspectral data, including:
    - reading fitted elemental maps and detector-element images,
    - basic visualisation helpers (line profiles, spectra, map overlays),
    - helpers for constructing weighted-sum maps.

  - `setkafluo/denoise.py`  
    Implementation of the Noise2Noise U-Net and training/inference helpers, including:
    - model construction and configuration,
    - creation of training datasets from detector-element splits,
    - training loops and logging utilities,
    - tiled prediction functions for large XRF maps.

  - `setkafluo/denoise_benchmark.py`  
    Utilities for timing and benchmarking denoising runs, e.g.:
    - measuring throughput for different patch/stride settings,
    - simple wrappers to reproduce the runtime comparisons in the paper.

- **Jupyter notebooks – end-to-end examples**

  The notebooks demonstrate how to use the library functions in practical workflows
  and reproduce the main analyses from the paper:

  - `01_data_exploration.ipynb`  
    Load fitted XRF maps, inspect detector-element images and spectra,
    and explore basic visualisation options.

  - `02_denoising_prep.ipynb`  
    Build Noise2Noise training pairs by splitting detector-element maps into
    two independent groups, extract patch datasets, and prepare input/target tensors.

  - `03_denoising_params.ipynb`  
    Experiment with training hyperparameters (patch size, batch size, learning rate,
    detector-element grouping) and inspect their effect on convergence and metrics.

  - `04_denoising_main.ipynb`  
    Run the main training workflow for the U-Net denoiser on Siemens star
    and/or cell datasets, saving trained models and logs.

  - `05_denoising_compare.ipynb`  
    Apply trained models to low-dose maps, compare against classical denoisers
    and high-dose references, and reproduce the quantitative metrics and figures
    reported in the paper.

---

## Versions

The code used for the paper is tagged `v0.1.0-paper` and archived at
https://doi.org/10.5281/zenodo.22282670. `main` is the maintained version (see
`CHANGELOG.md`). To reproduce the paper exactly: `git checkout v0.1.0-paper`.

What changed in `v0.2.0`:
- Package renamed `libs` → `setkafluo`; pip-installable (`pip install -e .`).
- Standardization rewritten as a plain per-image z-score (numerically identical; returned `std` is now σ rather than σ/μ).
- Augmentation now samples all 8 square orientations uniformly (was 6, non-uniform).
- Removed the PSNR monitoring metric (invalid on z-scored data; never affected training or published results).
- Removed unused architecture options (`dropout`, `instancenorm`, `averagepool`, `upconv`, `residual`, `lambda_reg`, `reg_l1`) and the custom `nullcontext`.
- Added docstrings with array shapes and a `pytest` test suite. The U-Net layer sequence is unchanged, so `v0.1.0-paper` weights load unchanged.

---

## Advanced

- **Mixed precision.** The model is mixed-precision ready but runs in float32 by
  default (as in the paper). To enable float16 compute on GPUs with tensor cores,
  set the global Keras policy before building the model —
  `from tensorflow.keras import mixed_precision; mixed_precision.set_global_policy("mixed_float16")` —
  then call `train()` as usual. The output layer is cast back to float32
  automatically and Keras applies loss scaling to the optimizer; no other changes
  are needed.

---

## Data and external resources

- **Article (Analytical Chemistry)** – full method and evaluation:  
  https://doi.org/10.1021/acs.analchem.5c05552

- **Preprint (ChemRxiv)** – open-access version:  
  https://doi.org/10.26434/chemrxiv-2025-lsxpc

- **Public dataset (Zenodo)** – Siemens star and human cancer cell XRF data used
  in the paper:  
  https://doi.org/10.5281/zenodo.17871605

Please consult the article for detailed information about sample preparation,
acquisition parameters, and preprocessing.

---

## Authors

This repository is jointly developed and maintained by:

- **Rodion Shishkov** – main developer. ESRF, Université Grenoble Alpes (UGA).  
- **Dmitry Karpov** – co-developer and supervising contributor. CEA / IRIG-MEM, Université Grenoble Alpes (UGA), ESRF.  

Other scientific contributors are listed in the article.

---

## License

This project is distributed under the **Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC-4.0)**.

You are free to:

- **Share** — copy and redistribute the material in any medium or format  
- **Adapt** — remix, transform, and build upon the material  

Under the following terms:

- **Attribution** — you must give appropriate credit and provide a link to the license.  
- **NonCommercial** — commercial use is strictly prohibited without prior written permission from the authors and ESRF.  

See the `LICENSE` file for the full legal text.

---

## How to cite

If you use this code in your work, please cite:

1. The article (and preprint):

> R. Shishkov, A. Laugros, N. Vigano, S. Bohic, D. Karpov, P. Cloetens  
> *Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy with Multi-Element Detectors*  
> Analytical Chemistry (2026), DOI: [10.1021/acs.analchem.5c05552](https://doi.org/10.1021/acs.analchem.5c05552)  
> Preprint: ChemRxiv (2025), DOI: [10.26434/chemrxiv-2025-lsxpc](https://doi.org/10.26434/chemrxiv-2025-lsxpc)

2. This code repository:

> R. Shishkov and D. Karpov. *SetkaFluo: Noise2Noise Denoising for XRF with Multi-Element Detectors* (software), version `v0.1.0-paper`. Zenodo (2026). https://doi.org/10.5281/zenodo.22282670

For the latest version use the concept DOI https://doi.org/10.5281/zenodo.22282669.
