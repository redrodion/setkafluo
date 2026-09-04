# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-09-04

### Changed
- Package renamed `libs` → `setkafluo`; now pip-installable (`pip install -e .`).
- Standardization rewritten as a plain per-image z-score (numerically identical;
  the returned `std` is now σ rather than σ/μ).
- Augmentation now samples all 8 square orientations uniformly (was 6, non-uniform).

### Removed
- PSNR monitoring metric (invalid on z-scored data; monitoring-only, never
  affected training or any published result).
- Unused architecture options (`dropout`, `instancenorm`, `averagepool`,
  `upconv`, `residual`, `lambda_reg`, `reg_l1`).
- Custom `nullcontext` (now `contextlib.nullcontext`).

### Added
- Docstrings with explicit array shapes and dtypes.
- Test suite (`pytest`).

The U-Net layer sequence is unchanged, so weights saved with `v0.1.0-paper` load
unchanged.

## [0.1.0-paper]

Code as used for the *Analytical Chemistry* paper. Tagged
[`v0.1.0-paper`](https://github.com/redrodion/setkafluo/releases/tag/v0.1.0-paper)
and archived on Zenodo at https://doi.org/10.5281/zenodo.22282670.
