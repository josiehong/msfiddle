# msfiddle

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/msfiddle)](https://pypi.org/project/msfiddle/)
[![Documentation](https://readthedocs.org/projects/msfiddle/badge/?version=latest)](https://msfiddle.readthedocs.io)

`msfiddle` is the PyPI package for FIDDLE, a deep learning method for chemical 
formula prediction from tandem mass spectra (MS/MS).

## Highlights

* Predict molecular formulas from MS/MS spectra with pre-trained FIDDLE models.
* Use the package from the command line, from native Python arrays, or from MGF files.
* Reuse loaded models for efficient batched prediction in Python applications.
* Incorporate BUDDY and SIRIUS candidate outputs in file-based workflows.

Paper: https://www.nature.com/articles/s41467-025-66060-9

Documentation: https://msfiddle.readthedocs.io

For the full experimental codebase, see https://github.com/JosieHong/FIDDLE.

## Installation

```bash
pip install msfiddle
```

PyTorch is required for inference. Install the optional inference extra, 
or install PyTorch separately for your platform:
```bash
pip install "msfiddle[inference]"
```

See the official PyTorch installation guide for custom CUDA builds:
https://pytorch.org/get-started/locally/.

## Usage

### Command-line interface

Download the pre-trained checkpoints before running predictions:
```bash
# Download models to the default location (~/.msfiddle/check_point)
msfiddle-download-models

# Or specify a custom location and models
msfiddle-download-models --destination /path/to/models \
                          --models fiddle_tcn_qtof fiddle_rescore_qtof
```

All `msfiddle` 2.x releases reuse the FIDDLE `v2.0.0` checkpoint assets.

Run the packaged demo:
```bash
msfiddle --demo --result_path ./output_demo.csv --device 0
```

Run the demo on CPU:
```bash
msfiddle --demo --result_path ./output_demo.csv --device 0 --no_cuda
```

Run prediction on your own [MGF file](#mgf-input):
```bash
msfiddle --test_data /path/to/data.mgf \
         --instrument_type orbitrap \
         --result_path /path/to/results.csv \
         --device 0
```

`--instrument_type` accepts `orbitrap` (default) or `qtof`. If checkpoints are
missing, the CLI exits with instructions to run `msfiddle-download-models`.

### Python API

Use `predict_from_spectrum` for one-off prediction from native MS/MS arrays:

```python
from msfiddle import predict_from_spectrum

candidates = predict_from_spectrum(
    mz_array=[60.0, 85.0, 100.0, 125.0, 150.0],
    intensity_array=[10.0, 50.0, 20.0, 35.0, 15.0],
    precursor_mz=180.063,
    adduct="[M+H]+",
    top_k=5,
    instrument_type="orbitrap",
    collision_energy="Unknown",
    device="cpu",
)
```

For repeated or batched prediction, reuse `MsFiddlePredictor` so checkpoints are
loaded once:
```python
from msfiddle import MsFiddlePredictor

predictor = MsFiddlePredictor(instrument_type="orbitrap", device="cpu")

results = predictor.predict_batch(
    [
        {
            "id": "sample-1",
            "mz_array": [60.0, 85.0, 100.0, 125.0, 150.0],
            "intensity_array": [10.0, 50.0, 20.0, 35.0, 15.0],
            "precursor_mz": 180.063,
            "adduct": "[M+H]+",
            "collision_energy": "Unknown",
        }
    ]
)
```

Python APIs do not download model checkpoints unless `download_models=True` is passed.

## Input and output formats

### CSV output

The CLI writes a CSV file with one row per spectrum. Key columns include:

| Column | Description |
| --- | --- |
| `ID` | Spectrum title from the MGF file. |
| `Mass` | Neutral mass calculated from precursor m/z and adduct. |
| `Pred Formula` | Initial formula predicted by the neural model. |
| `Pred Mass` | Model-predicted mass. |
| `Pred Atom Num` | Model-predicted atom count. |
| `Pred H/C Num` | Model-predicted H/C count. |
| `Refined Formula (0..4)` | Ranked refined formula candidates for the default top-5 output. |
| `Refined Mass (0..4)` | Masses for the default top-5 refined candidates. |
| `Rescore (0..4)` | Confidence scores for the default top-5 refined candidates. |

### API output

The Python `predict_from_spectrum()` API returns a list of candidate dictionaries:
```python
[
    {
        "formula": "C8H10O",
        "score": 0.94,
        "mass": 122.073,
        "metadata": {...},
    }
]
```

`predict_batch()` returns one record per input spectrum with `id`, `candidates`,
and `metadata`.

### Native/original BUDDY and SIRIUS outputs

Use the same MGF input file to generate native
[BUDDY/msbuddy](https://msbuddy.readthedocs.io/en/latest/cmdapi.html) and
[SIRIUS](https://v6.docs.sirius-ms.io/cli/) outputs. `msfiddle` accepts these
native/original files directly through `--buddy_path` and `--sirius_path`. The
older msfiddle-normalized CSV formats documented in `docs/formats.rst` are
**deprecated** and will be removed in `msfiddle` 3.0.0; loading them emits a
`DeprecationWarning`.

```bash
msbuddy -mgf /path/to/data.mgf \
        -output /path/to/buddy_output \
        -ms orbitrap \
        -p -n_cpu 12 \
        -d -hal
```

`msbuddy` writes `msbuddy_result_summary.tsv`; `-d` also writes detailed
per-query `formula_results.tsv` files. Pass either the summary file or the
output directory to `--buddy_path`. The full output directory is preferred
because it includes per-candidate FDR scores for ranks beyond rank 1.

```bash
sirius --input /path/to/data.mgf \
       --project /path/to/sirius_project \
       formulas --profile orbitrap

sirius --project /path/to/sirius_project \
       summaries --top-k-summary=5 \
       --output /path/to/sirius_output
```

SIRIUS writes a project space and exports summary files such as
`formula_identifications.tsv`. Pass either a formula-identification summary file
or the summary output directory to `--sirius_path`.
SIRIUS 6 may require `sirius login` before formula computation.

### MGF input

The required MGF fields are `TITLE`, `PRECURSOR_MZ`, `PRECURSOR_TYPE`, and
`COLLISION_ENERGY`:
```mgf
BEGIN IONS
TITLE=EMBL_MCF_2_0_HRMS_Library000529
PEPMASS=111.02016
CHARGE=1-
PRECURSOR_TYPE=[M-H]-
PRECURSOR_MZ=111.02016
COLLISION_ENERGY=50.0
SMILES=[H]c1c([H])n([H])c(=O)n([H])c1=O
FORMULA=C4H4N2O2
THEORETICAL_PRECURSOR_MZ=111.019453
PPM=6.368253318682487
SIMULATED_PRECURSOR_MZ=111.01946768634916
41.0148 0.329893 
41.9986 89.226766 
55.8055 0.200544 
56.2625 0.194617 
67.0304 0.330612 
68.0258 0.402906 
111.0203 100.0 
112.0515 1.2809 
END IONS
```

## Advanced Usage

Inspect checkpoint paths:
```bash
msfiddle-checkpoint-paths
```

Use custom config and checkpoint paths:
```bash
msfiddle --test_data /path/to/data.mgf \
         --config_path /path/to/config.yml \
         --resume_path /path/to/tcn_model.pt \
         --rescore_resume_path /path/to/rescore_model.pt \
         --result_path /path/to/results.csv \
         --device 0
```

## Citation

```
@article{hong2025fiddle,
  title={FIDDLE: a deep learning method for chemical formulas prediction from tandem mass spectra},
  author={Hong, Yuhui and Li, Sujun and Ye, Yuzhen and Tang, Haixu},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={11102},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
