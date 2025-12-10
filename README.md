# SetkaFluo: a Multi-Element Detector Denoising Framework

Tired of averaging your multi-element detector data and pretending that's the best we can do? Convinced there must be a smarter way? **SetkaFluo** is exactly that: a small library built to squeeze more information out of the data you already collect.

Using the Noise2Noise framework, SetkaFluo takes advantage of what your experiment naturally provides: repeated, independent noisy observations of the same underlying signal. Whether you're working with XRF or any modality where multiple detector elements observe the same spot, the structure is consistent and only the noise changes. That's exactly what we exploit.

A U-Net architecture handles the heavy lifting. Instead of blurring or smudging away your details, the network learns to separate actual structure from randomness, restoring that crisp, "salt-free" look without supervision, ground truth, or elaborate parameter tuning.

The library is designed to work out-of-the-box on Google Colab, including on free-tier machines (a T4 may need a coffee break or two, but it gets the job done).

To help you get started quickly, we include step-by-step Jupyter Notebooks that introduce the ideas, workflow, and practical details. The notebooks contain all essential explanations, so reading the comments is highly recommended.

## Publication

This package is the core implementation of our manuscript currently available as a preprint:

**Shishkov R, Laugros A, Vigano N, Bohic S, Karpov D ✉, Cloetens P ✉.** Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy with Multi-Element Detectors. *ChemRxiv*. 2025; doi:[10.26434/chemrxiv-2025-lsxpc](https://doi.org/10.26434/chemrxiv-2025-lsxpc)

*This content is a preprint and is currently in peer-review.*

---

## Main contributors

This repository is jointly developed and maintained by:

- **Rodion Shishkov** – main developer (ESRF, UGA)  
- **Dmitry Karpov** – co-developer and supervising contributor (CEA, UGA, ESRF)  

Other scientific contributors are listed in the preprint.

---

## How to Test-Drive This Library Through Dedicated Jupyter Notebooks Using Google Colab

The fastest way to explore **SetkaFluo** is to run the tutorial notebooks directly in Google Colab, using the example datasets published on Zenodo.

This setup requires **no local installation**, **no GPU configuration**, and **no manual environment management**. A free Google account is sufficient.

### 1. Download the Example Dataset (Zenodo)

The data needed for the notebooks are provided in the Zenodo archive:

**https://doi.org/10.5281/zenodo.17871605**

It contains two files:
- `input_data.zip` — fitted XRF maps and detector-element images
- `training.zip` — detector-element stacks for constructing Noise2Noise training pairs

These datasets correspond to the preprint:

> Shishkov R, Laugros A, Vigano N, Bohic S, Karpov D, Cloetens P.  
> *Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy with Multi-Element Detectors.*  
> ChemRxiv (2025). doi:[10.26434/chemrxiv-2025-lsxpc](https://doi.org/10.26434/chemrxiv-2025-lsxpc)

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

### 3. Add the Notebooks and Library Code

From this GitHub repository, copy:
- the five tutorial notebooks, and
- the `libs/` directory

into:
```
MyDrive/setkafluo_demo/notebooks_and_library/
    01_data_exploration.ipynb
    02_denoising_prep.ipynb
    03_denoising_params.ipynb
    04_denoising_main.ipynb
    05_denoising_compare.ipynb
    libs/
```

This reproduces the environment used to generate the figures and benchmarks shown in the paper and preprint.

### 4. Run the Notebooks in Google Colab

Open any of the notebooks through Colab.

All paths are already configured to expect the dataset under:
```
MyDrive/setkafluo_demo/
```

The notebooks will run without modification on standard Colab GPU runtimes.

### Local Installation

If you prefer to run the library on your own workstation (Python environment + TensorFlow), see:

➡ **[Jump to: Repository Overview & Installation](#repository-overview)**

This section provides full instructions for creating a dedicated environment and installing all dependencies.

---

## Repository overview

The repository is organized in two main parts:

- **`libs/` – core library**

  This folder contains the reusable Python code that implements the SetkaFluo pipeline.
  It can be imported from your own scripts or used directly in the notebooks.

  - `libs/data_explorer.py`  
    Utilities for loading and exploring XRF hyperspectral data, including:
    - reading fitted elemental maps and detector-element images,
    - basic visualisation helpers (line profiles, spectra, map overlays),
    - helpers for constructing weighted-sum maps.

  - `libs/denoise.py`  
    Implementation of the Noise2Noise U-Net and training/inference helpers, including:
    - model construction and configuration,
    - creation of training datasets from detector-element splits,
    - training loops and logging utilities,
    - tiled prediction functions for large XRF maps.

  - `libs/denoise_benchmark.py`  
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

## Installation

We strongly recommend using a **dedicated virtual environment** for this project.

TensorFlow is the core dependency and should be installed following the official
instructions for your OS and hardware (CPU/GPU). For this reason, **TensorFlow is
not included in `requirements.txt` or `environment.yml`** – you install it first,
then install the remaining dependencies.

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

### 2. Clone this repository

With the virtual environment **activated**:
```bash
git clone https://github.com/redrodion/setkafluo.git
cd setkafluo
```

### 3. Install Python dependencies

There are two equivalent options:

#### Option A – `pip` + `requirements.txt` (simple)
```bash
pip install -r requirements.txt
```

This installs all dependencies **except TensorFlow**, which you already installed
in step 1.

#### Option B – `conda` + `environment.yml` (if you prefer conda)

You can create a conda environment pre-populated with the non-TensorFlow
dependencies:
```bash
conda env create -f environment.yml
conda activate setkafluo
```

Then, inside this conda environment, install TensorFlow using `pip` as described
in step 1 (following the official TensorFlow guide for your platform/GPU).

---

## How to use

Once the environment is set up and dependencies are installed:

1. **Launch Jupyter**

   From the activated environment:
```bash
   jupyter lab
   # or
   jupyter notebook
```

2. **Run the notebooks in order**

   - Start with `01_data_exploration.ipynb` to familiarise yourself with the data.
   - Use `02_denoising_prep.ipynb` to generate Noise2Noise training patches.
   - Adjust hyperparameters in `03_denoising_params.ipynb` if desired.
   - Train the model in `04_denoising_main.ipynb`.
   - Evaluate and compare results in `05_denoising_compare.ipynb`.

3. **Use the library in your own scripts**

   You can also import the core functions directly:
```python
   from libs.denoise import (
       make_unet,
       make_dataset,
       train,
       predict_tiled,
   )

   from libs.data_explorer import (
       load_npz_cube_channels_last,
       sum_channels_window,
   )
```

   See the notebooks for concrete examples of how to construct training pairs,
   configure the model, and run inference on large XRF maps.

---

## Data and external resources

- **Preprint (ChemRxiv)** – full method and evaluation:  
  https://doi.org/10.26434/chemrxiv-2025-lsxpc 

- **Public dataset (Zenodo)** – Siemens star and human cancer cell XRF data used
  in the paper:  
  https://doi.org/10.5281/zenodo.17871605

Please consult the preprint for detailed information about sample preparation,
acquisition parameters, and preprocessing.

---

## Authors and affiliations

Main contributors of this repository:

- **Rodion Shishkov** – ESRF, Université Grenoble Alpes (UGA)  
- **Dmitry Karpov** – CEA / IRIG-MEM, Université Grenoble Alpes (UGA), ESRF  

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

1. The preprint / article:

> R. Shishkov, A. Laugros, N. Vigano, S. Bohic, D. Karpov, P. Cloetens  
> *Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy with Multi-Element Detectors*  
> ChemRxiv (2025), DOI: [10.26434/chemrxiv-2025-lsxpc](https://doi.org/10.26434/chemrxiv-2025-lsxpc)

2. This code repository (GitHub URL and version / commit hash), e.g.:

> R. Shishkov and D. Karpov  
> *SetkaFluo: Noise2Noise Denoising for XRF with Multi-Element Detectors (code repository)*  
> GitHub, https://github.com/redrodion/setkafluo (accessed YYYY-MM-DD)