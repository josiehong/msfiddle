# Changelog

All notable changes to `msfiddle` will be documented in this file.

## [2.0.1] - 2026-05-02

### Added
- Added `MsFiddlePredictor` for reusable Python inference with single-spectrum, batch, and MGF prediction methods.
- Added `predict_from_spectrum`, `predict_batch_from_spectra`, and `predict_from_mgf` convenience APIs.
- Added the optional `inference` extra for installing PyTorch with `pip install "msfiddle[inference]"`.

### Changed
- Refactored the CLI to use the shared predictor internals while preserving the existing command-line interface and CSV output shape.
- Deferred checkpoint warnings/errors until prediction instead of warning during import.
- Set package metadata to require Python 3.8+, matching the existing pandas 2 dependency.

## [2.0.0] - 2026-03-23

### Changed
- Replaced `FDRNet` with a Siamese-style rescoring architecture: new `FormulaEncoder` (MLP → L2-normalised embedding) and `RescoreHead` (element-wise product → scalar logit) classes in `model_tcn.py`
- Renamed `FDRDataset` → `RescoreDataset` in `dataset.py` and updated references from `prepare_fdr.py` to `prepare_rescore.py`
- Renamed `train_fdr` config section to `train_rescore` across all four config YAMLs
- Reduced `early_stop_step` from 10 to 5 in Orbitrap and Q-TOF training configs

### Added
- `formula_dim: 64` parameter added to Orbitrap and Q-TOF model configs

## [0.1.0] - 2025-03-20

### Added
- Initial release
- Chemical formula prediction from tandem mass spectra (MS/MS) using pre-trained TCN models
- Support for Orbitrap and Q-TOF instrument types
- Formula refinement with confidence scoring (FDR)
- Integration with BUDDY and SIRIUS results
- `msfiddle` CLI for running predictions
- `msfiddle-download-models` CLI for downloading pre-trained model weights
- `msfiddle-checkpoint-paths` CLI for inspecting model locations
- Demo data for quick testing (`--demo` flag)
