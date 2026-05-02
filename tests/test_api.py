import pytest

import msfiddle.api as api
from msfiddle.api import MsFiddlePredictor, _ModelPrediction


VALID_MZ = [60.0, 85.0, 100.0, 125.0, 150.0]
VALID_INTENSITY = [10.0, 50.0, 20.0, 35.0, 15.0]
VALID_PRECURSOR_MZ = 180.063
VALID_ADDUCT = "[M+H]+"


def test_predict_from_spectrum_rejects_unequal_array_lengths():
    with pytest.raises(ValueError, match="same length"):
        api.predict_from_spectrum(
            mz_array=VALID_MZ,
            intensity_array=VALID_INTENSITY[:-1],
            precursor_mz=VALID_PRECURSOR_MZ,
            adduct=VALID_ADDUCT,
        )


def test_predict_from_spectrum_rejects_too_few_peaks():
    with pytest.raises(ValueError, match="at least 5 peaks"):
        api.predict_from_spectrum(
            mz_array=VALID_MZ[:4],
            intensity_array=VALID_INTENSITY[:4],
            precursor_mz=VALID_PRECURSOR_MZ,
            adduct=VALID_ADDUCT,
        )


def test_predict_from_spectrum_rejects_unsupported_adduct():
    with pytest.raises(ValueError, match="Unsupported adduct"):
        api.predict_from_spectrum(
            mz_array=VALID_MZ,
            intensity_array=VALID_INTENSITY,
            precursor_mz=VALID_PRECURSOR_MZ,
            adduct="[M+K]+",
        )


def test_predict_batch_rejects_malformed_item():
    with pytest.raises(ValueError, match="missing 'intensity_array'"):
        api.predict_batch_from_spectra(
            [
                {
                    "mz_array": VALID_MZ,
                    "precursor_mz": VALID_PRECURSOR_MZ,
                    "adduct": VALID_ADDUCT,
                }
            ]
        )


def test_predict_from_spectrum_reports_missing_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing msfiddle checkpoint"):
        api.predict_from_spectrum(
            mz_array=VALID_MZ,
            intensity_array=VALID_INTENSITY,
            precursor_mz=VALID_PRECURSOR_MZ,
            adduct=VALID_ADDUCT,
            resume_path=tmp_path / "missing_tcn.pt",
            rescore_resume_path=tmp_path / "missing_rescore.pt",
        )


def test_predict_from_spectrum_delegates_to_predictor(monkeypatch):
    calls = {}

    class DummyPredictor:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def predict_spectrum(self, *args, **kwargs):
            calls["predict"] = {"args": args, "kwargs": kwargs}
            return [
                {
                    "formula": "C8H9N3S",
                    "score": 0.9,
                    "mass": 179.05,
                    "metadata": {"raw_rescore": 0.9},
                }
            ]

    monkeypatch.setattr(api, "MsFiddlePredictor", DummyPredictor)

    result = api.predict_from_spectrum(
        mz_array=VALID_MZ,
        intensity_array=VALID_INTENSITY,
        precursor_mz=VALID_PRECURSOR_MZ,
        adduct=VALID_ADDUCT,
        device="cpu",
        download_models=False,
    )

    assert calls["init"]["device"] == "cpu"
    assert calls["init"]["download_models"] is False
    assert calls["predict"]["kwargs"]["top_k"] == 5
    assert result[0]["formula"] == "C8H9N3S"


def test_predict_batch_preserves_order_ids_and_candidate_shape(monkeypatch, tmp_path):
    predictor = MsFiddlePredictor(
        resume_path=tmp_path / "missing_tcn.pt",
        rescore_resume_path=tmp_path / "missing_rescore.pt",
    )

    def fake_predict_encoded(self, encoded, *, batch_size):
        return [
            _ModelPrediction(
                spectrum=item,
                vector_string="1;2;3",
                pred_formula="C8H10O",
                pred_mass=122.1,
                pred_atom_count=19.0,
                pred_hc_count=1.25,
                prediction_time=0.01,
            )
            for item in encoded
        ]

    def fake_refine_and_rescore(self, prediction, top_k, *, extra_formulas=None):
        return (
            {
                "formula": ["C8H10O", None],
                "mass": [122.073, None],
                "rescore": [1.2, 0.0],
            },
            122.073,
        )

    monkeypatch.setattr(MsFiddlePredictor, "_predict_encoded", fake_predict_encoded)
    monkeypatch.setattr(MsFiddlePredictor, "_refine_and_rescore", fake_refine_and_rescore)

    result = predictor.predict_batch(
        [
            {
                "id": "a",
                "mz_array": VALID_MZ,
                "intensity_array": VALID_INTENSITY,
                "precursor_mz": VALID_PRECURSOR_MZ,
                "adduct": VALID_ADDUCT,
            },
            {
                "id": "b",
                "mz_array": [70.0, 90.0, 110.0, 130.0, 160.0],
                "intensity_array": VALID_INTENSITY,
                "precursor_mz": 190.063,
                "adduct": VALID_ADDUCT,
            },
        ],
        top_k=2,
    )

    assert [record["id"] for record in result] == ["a", "b"]
    assert result[0]["candidates"][0]["formula"] == "C8H10O"
    assert result[0]["candidates"][0]["score"] == 1.0
    assert result[0]["candidates"][0]["mass"] == pytest.approx(122.073)
    assert result[0]["candidates"][0]["metadata"]["raw_rescore"] == 1.2
    assert result[0]["metadata"]["adduct"] == VALID_ADDUCT
