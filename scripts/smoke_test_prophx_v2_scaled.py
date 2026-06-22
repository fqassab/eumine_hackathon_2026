from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from pymatgen.core import Structure


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "prophx_v2_scaled"
SOURCE_DIR = ROOT / "src" / "prophx_v2_scaled"

VALIDATION_ZIP = (
    ROOT
    / "data"
    / "raw"
    / "eumine"
    / "bridge_dataset_val_structures.zip"
)


def main() -> None:
    required_paths = [
        MODEL_DIR / "physics_model_ef.joblib",
        MODEL_DIR / "physics_model_bg.joblib",
        MODEL_DIR / "feature_columns.joblib",
        MODEL_DIR / "metadata.joblib",
        MODEL_DIR / "discrepancy_lookup.joblib",
        SOURCE_DIR / "ProphX_Predictor.py",
        SOURCE_DIR / "scaled_model_wrapper.py",
        VALIDATION_ZIP,
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n- " + "\n- ".join(missing)
        )

    sys.path.insert(0, str(SOURCE_DIR))

    # This must load before joblib loads the scaled formation-energy model.
    import scaled_model_wrapper  # noqa: F401
    from ProphX_Predictor import ProphXPredictor

    with zipfile.ZipFile(VALIDATION_ZIP, "r") as archive:
        cif_files = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".cif")
        )

        if not cif_files:
            raise RuntimeError("No CIF files found in validation ZIP.")

        cif_name = cif_files[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_cif = Path(temp_dir) / Path(cif_name).name

            with archive.open(cif_name, "r") as source:
                temporary_cif.write_bytes(source.read())

            structure = Structure.from_file(temporary_cif)

    predictor = ProphXPredictor()
    predictor.load_model(str(MODEL_DIR))

    outputs = predictor.predict([structure])

    if len(outputs) != 1:
        raise RuntimeError(
            f"Expected 1 prediction, received {len(outputs)}."
        )

    result = outputs[0]

    required_keys = [
        "formation_energy_per_atom",
        "band_gap",
        "model_id",
        "data_sources_used",
    ]

    missing_keys = [
        key for key in required_keys if key not in result
    ]

    if missing_keys:
        raise RuntimeError(
            f"Prediction is missing required keys: {missing_keys}"
        )

    formation_energy = float(result["formation_energy_per_atom"])
    band_gap = float(result["band_gap"])

    if not np.isfinite(formation_energy):
        raise RuntimeError("Formation-energy prediction is not finite.")

    if not np.isfinite(band_gap):
        raise RuntimeError("Band-gap prediction is not finite.")

    print("=" * 72)
    print("ProphX_v2_scaled smoke test passed")
    print("=" * 72)
    print("Input CIF:", cif_name)
    print("Model ID:", result["model_id"])
    print("Data sources:", result["data_sources_used"])
    print(f"Formation energy per atom: {formation_energy:.6f}")
    print(f"Band gap: {band_gap:.6f}")


if __name__ == "__main__":
    main()