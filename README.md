# SetkaFluo – Noise2Noise denoising for XRF with multi-element detectors

This repository contains the reference implementation of the **SetkaFluo** framework
for self-supervised denoising of X-ray fluorescence (XRF) microscopy data acquired
with multi-element detectors. The method and results are described in:

> R. Shishkov, A. Laugros, N. Vigano, S. Bohic, D. Karpov, P. Cloetens  
> *Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy with Multi-Element Detectors*  
> ChemRxiv (2025), DOI: [10.26434/chemrxiv-2025-lsxpc](https://doi.org/10.26434/chemrxiv-2025-lsxpc)

Please refer to the preprint for the full scientific context, evaluation, and figures.

---

## Main contributors

This repository is jointly developed and maintained by:

- **Rodion Shishkov** – main developer (ESRF, UGA)  
- **Dmitry Karpov** – co-developer and supervising contributor (CEA, UGA, ESRF)  

Other scientific contributors are listed in the preprint.

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
  https://chemrxiv.org/engage/chemrxiv/article-details/6899b33f728bf9025ef81f5a  

- **Public dataset (Zenodo)** – Siemens star and human cancer cell XRF data used
  in the paper:  
  _[Zenodo link to be added by authors]_  

- **Google Colab demo** – online notebook that sets up the environment, downloads
  example data, and runs a minimal denoising workflow:  
  _[Colab link to be added by authors]_  

Please consult the preprint for detailed information about sample preparation,
acquisition parameters, and preprocessing.

---

## Authors and affiliations

Main contributors of this repository:

- **Rodion Shishkov** – ESRF, Université Grenoble Alpes (UGA)  
- **Dmitry Karpov** – CEA / IRIG-MEM, Université Grenoble Alpes (UGA), ESRF  

Contact (to be finalised):

- Primary contact: _[Rodion email to be added]_  
- Secondary contact: _[Dmitry email to be added]_  

---

## License

This project is distributed under the **MIT License**.

See the `LICENSE` file for the full text. Please check your institutional policies
before using the code in commercial settings.

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
