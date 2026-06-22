from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from matminer.featurizers.composition import (
    ElementProperty,
    IonProperty,
    OxidationStates,
    Stoichiometry,
)
from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import Element

try:
    from matfed_api import MatFedPredictor
except Exception:
    # Local fallback for Jupyter/testing when matfed_api is not available.
    # In the official EuMINe/MatFed environment, the real MatFedPredictor will be used.
    class MatFedPredictor:
        pass


class ProphXPredictor(MatFedPredictor):
    """
    EuMINe / MatFed-compatible predictor.

    Required API methods:
        load_model(self, model_path: str) -> None
        predict(self, structures: List[Structure]) -> List[Dict]
        describe(self) -> Dict

    Model:
        - Feature extraction from pymatgen Structure objects
        - MAGPIE + stoichiometry + oxidation + ionic + electronegativity features
        - Structure-derived features
        - Two separate regressors:
            1. formation_energy_per_atom
            2. band_gap

    Discrepancy-aware behavior:
        - During training, MP/EuMINe and JARVIS-DFT discrepancies are used as sample-confidence weights.
        - At prediction time, if the input structure matches a known training/matched entry,
          source values and discrepancies are reported.
        - If no match is found, the output explains that discrepancy was used during training
          but cannot be recomputed for this unknown structure.
    """

    def __init__(self) -> None:
        self.magpie = ElementProperty.from_preset("magpie")
        self.stoich = Stoichiometry()
        self.oxi = OxidationStates()
        self.ion = IonProperty(fast=True)

        self.model_ef = None
        self.model_bg = None
        self.feature_columns = None
        self.discrepancy_lookup = {}
        self.metadata: Dict[str, Any] = {}

        self.model_id = "prophx_physics_guided_magpie_structure_v1"
        self.data_sources_used = ["EuMINe", "AFLOW", "JARVIS-DFT"]

    def load_model(self, model_path: str) -> None:
        """
        Load saved model artifacts from disk.

        Expected files inside model_path:
            physics_model_ef.joblib
            physics_model_bg.joblib
            feature_columns.joblib
            metadata.joblib
            discrepancy_lookup.joblib
        """
        model_dir = Path(model_path)

        ef_path = model_dir / "physics_model_ef.joblib"
        bg_path = model_dir / "physics_model_bg.joblib"
        columns_path = model_dir / "feature_columns.joblib"
        metadata_path = model_dir / "metadata.joblib"
        lookup_path = model_dir / "discrepancy_lookup.joblib"

        if ef_path.exists() and bg_path.exists():
            self.model_ef = joblib.load(ef_path)
            self.model_bg = joblib.load(bg_path)

        if columns_path.exists():
            self.feature_columns = joblib.load(columns_path)

        if metadata_path.exists():
            try:
                self.metadata = joblib.load(metadata_path)
                self.model_id = self.metadata.get("model_id", self.model_id)
                self.data_sources_used = self.metadata.get(
                    "data_sources_used",
                    self.data_sources_used,
                )
            except Exception:
                self.metadata = {}

        if lookup_path.exists():
            try:
                self.discrepancy_lookup = joblib.load(lookup_path)
            except Exception:
                self.discrepancy_lookup = {}

        return None

    def _safe_float(self, value, default=np.nan) -> float:
        try:
            if value is None:
                return default

            value = float(value)

            if np.isfinite(value):
                return value

            return default

        except Exception:
            return default

    def _weighted_average_electronegativity(self, comp: Composition) -> float:
        values = []
        weights = []

        for el, amount in comp.get_el_amt_dict().items():
            try:
                x = Element(el).X

                if x is not None:
                    values.append(float(x))
                    weights.append(float(amount))

            except Exception:
                continue

        if not values:
            return np.nan

        return float(np.average(values, weights=weights))

    def _min_electronegativity_difference(self, comp: Composition) -> float:
        xs = []

        for el in comp.get_el_amt_dict().keys():
            try:
                x = Element(el).X

                if x is not None:
                    xs.append(float(x))

            except Exception:
                continue

        if len(xs) < 2:
            return 0.0

        diffs = []

        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                diffs.append(abs(xs[i] - xs[j]))

        if not diffs:
            return 0.0

        return float(min(diffs))

    def _average_abs_oxidation_state(self, comp: Composition) -> float:
        try:
            guesses = comp.oxi_state_guesses()

            if not guesses:
                return np.nan

            guess = guesses[0]
            total_atoms = sum(comp.get_el_amt_dict().values())

            weighted_sum = 0.0

            for el, amount in comp.get_el_amt_dict().items():
                oxi = guess.get(el, np.nan)
                oxi = float(oxi)

                if not np.isfinite(oxi):
                    return np.nan

                weighted_sum += abs(oxi) * float(amount)

            return float(weighted_sum / total_atoms)

        except Exception:
            return np.nan

    def _structure_features(self, structure: Structure) -> List[float]:
        try:
            nsites = len(structure)
            volume = structure.volume
            volume_per_atom = volume / nsites if nsites > 0 else np.nan
            inv_r = volume_per_atom ** (-1.0 / 3.0) if volume_per_atom > 0 else np.nan
            density = structure.density

        except Exception:
            nsites = np.nan
            volume = np.nan
            volume_per_atom = np.nan
            inv_r = np.nan
            density = np.nan

        try:
            lattice = structure.lattice
            a = lattice.a
            b = lattice.b
            c = lattice.c
            alpha = lattice.alpha
            beta = lattice.beta
            gamma = lattice.gamma

        except Exception:
            a = np.nan
            b = np.nan
            c = np.nan
            alpha = np.nan
            beta = np.nan
            gamma = np.nan

        return [
            self._safe_float(nsites),
            self._safe_float(volume),
            self._safe_float(volume_per_atom),
            self._safe_float(inv_r),
            self._safe_float(density),
            self._safe_float(a),
            self._safe_float(b),
            self._safe_float(c),
            self._safe_float(alpha),
            self._safe_float(beta),
            self._safe_float(gamma),
        ]

    def _material_class_features(self, comp: Composition) -> List[float]:
        elements = set(comp.get_el_amt_dict().keys())

        return [
            float("O" in elements),
            float("S" in elements),
            float("N" in elements),
            float(any(el in elements for el in ["F", "Cl", "Br", "I"])),
            float("P" in elements),
            float("C" in elements),
        ]

    def _feature_labels(self) -> List[str]:
        labels = []

        labels.extend([f"magpie::{x}" for x in self.magpie.feature_labels()])
        labels.extend([f"stoich::{x}" for x in self.stoich.feature_labels()])
        labels.extend([f"oxidation::{x}" for x in self.oxi.feature_labels()])
        labels.extend([f"ion::{x}" for x in self.ion.feature_labels()])

        labels.extend(
            [
                "custom::avg_electronegativity",
                "custom::min_electronegativity_difference",
                "custom::avg_abs_oxidation_state",
                "custom::n_elements",
            ]
        )

        labels.extend(
            [
                "structure::nsites",
                "structure::volume",
                "structure::volume_per_atom",
                "structure::inv_r_volume_per_atom",
                "structure::density",
                "structure::lattice_a",
                "structure::lattice_b",
                "structure::lattice_c",
                "structure::lattice_alpha",
                "structure::lattice_beta",
                "structure::lattice_gamma",
            ]
        )

        labels.extend(
            [
                "class::is_oxide",
                "class::is_sulfide",
                "class::is_nitride",
                "class::is_halide",
                "class::contains_phosphorus",
                "class::contains_carbon",
            ]
        )

        return labels

    def _featurize_one(self, structure: Structure) -> List[float]:
        comp = structure.composition.reduced_composition

        features = []

        try:
            features.extend(self.magpie.featurize(comp))
        except Exception:
            features.extend([np.nan] * len(self.magpie.feature_labels()))

        try:
            features.extend(self.stoich.featurize(comp))
        except Exception:
            features.extend([np.nan] * len(self.stoich.feature_labels()))

        try:
            features.extend(self.oxi.featurize(comp))
        except Exception:
            features.extend([np.nan] * len(self.oxi.feature_labels()))

        try:
            features.extend(self.ion.featurize(comp))
        except Exception:
            features.extend([np.nan] * len(self.ion.feature_labels()))

        avg_en = self._weighted_average_electronegativity(comp)
        min_en_diff = self._min_electronegativity_difference(comp)
        avg_abs_oxi = self._average_abs_oxidation_state(comp)
        n_elements = len(comp.elements)

        features.extend(
            [
                self._safe_float(avg_en),
                self._safe_float(min_en_diff),
                self._safe_float(avg_abs_oxi),
                self._safe_float(n_elements),
            ]
        )

        features.extend(self._structure_features(structure))
        features.extend(self._material_class_features(comp))

        return [self._safe_float(value) for value in features]

    def _featurize(self, structures: List[Structure]) -> np.ndarray:
        features = [self._featurize_one(structure) for structure in structures]
        x_values = np.asarray(features, dtype=float)

        if self.feature_columns is not None:
            expected = len(self.feature_columns)
            actual = x_values.shape[1]

            if actual < expected:
                padding = np.full((x_values.shape[0], expected - actual), np.nan)
                x_values = np.hstack([x_values, padding])

            elif actual > expected:
                x_values = x_values[:, :expected]

        return x_values

    def _structure_match_key(self, structure: Structure) -> str:
        formula = structure.composition.reduced_formula
        nsites = len(structure)

        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            sg = int(sga.get_space_group_number())
        except Exception:
            sg = None

        return f"{formula}|{int(nsites)}|{sg}"

    def _structure_backup_match_key(self, structure: Structure) -> str:
        formula = structure.composition.reduced_formula
        nsites = len(structure)

        return f"{formula}|{int(nsites)}|None"

    def _confidence_level_from_score(self, score) -> str:
        try:
            score = float(score)
        except Exception:
            return "medium"

        if score >= 0.80:
            return "high"
        if score >= 0.50:
            return "medium"
        return "low"

    def _lookup_discrepancy_info(self, structure: Structure) -> Dict:
        if not self.discrepancy_lookup:
            return {
                "discrepancy_available": False,
                "matched_database_entry": None,
                "source_values": None,
                "source_discrepancies": None,
                "confidence_score": 0.60,
                "confidence_level": "medium",
                "discrepancy_handling": (
                    "No discrepancy lookup was loaded. Cross-database discrepancies "
                    "may have been used during training, but no material-specific "
                    "source discrepancy can be reported for this prediction."
                ),
            }

        key = self._structure_match_key(structure)
        backup_key = self._structure_backup_match_key(structure)

        info = self.discrepancy_lookup.get(key)

        if info is None:
            info = self.discrepancy_lookup.get(backup_key)

        if info is None:
            return {
                "discrepancy_available": False,
                "matched_database_entry": None,
                "source_values": None,
                "source_discrepancies": None,
                "confidence_score": 0.60,
                "confidence_level": "medium",
                "discrepancy_handling": (
                    "No matching MP/JARVIS-DFT entry was found for this input structure. "
                    "The model prediction is structure-only; cross-database discrepancies "
                    "were used during training through sample-confidence weighting but "
                    "cannot be recomputed for this specific structure."
                ),
            }

        confidence_score = info.get("confidence_score", 0.60)
        confidence_level = info.get(
            "confidence_level",
            self._confidence_level_from_score(confidence_score),
        )

        return {
            "discrepancy_available": True,
            "matched_database_entry": {
                "material_id": info.get("material_id", None),
                "jid": info.get("jid", None),
                "formula": info.get("formula", None),
                "spacegroup_number": info.get("spacegroup_number", None),
                "nsites": info.get("nsites", None),
            },
            "source_values": {
                "formation_energy_per_atom_mp": self._safe_float(
                    info.get("formation_energy_per_atom_mp", np.nan)
                ),
                "formation_energy_per_atom_jarvis": self._safe_float(
                    info.get("formation_energy_per_atom_jarvis", np.nan)
                ),
                "band_gap_mp": self._safe_float(
                    info.get("band_gap_mp", np.nan)
                ),
                "band_gap_jarvis": self._safe_float(
                    info.get("band_gap_jarvis", np.nan)
                ),
            },
            "source_discrepancies": {
                "formation_energy_difference_mp_minus_jarvis": self._safe_float(
                    info.get("formation_energy_discrepancy", np.nan)
                ),
                "absolute_formation_energy_difference": self._safe_float(
                    info.get("abs_formation_energy_discrepancy", np.nan)
                ),
                "band_gap_difference_mp_minus_jarvis": self._safe_float(
                    info.get("band_gap_discrepancy", np.nan)
                ),
                "absolute_band_gap_difference": self._safe_float(
                    info.get("abs_band_gap_discrepancy", np.nan)
                ),
            },
            "confidence_score": self._safe_float(confidence_score, default=0.60),
            "confidence_level": str(confidence_level),
            "discrepancy_handling": (
                "This structure matched a known entry with both MP/EuMINe and "
                "JARVIS-DFT labels. Source-specific values and discrepancies are "
                "reported, and the confidence level reflects the size of the "
                "cross-database disagreement."
            ),
        }

    def predict(self, structures: List[Structure]) -> List[Dict]:
        """
        Predict formation energy per atom and band gap.

        Input:
            structures: List[pymatgen.core.Structure]

        Output:
            List[Dict], same length as input.

        Required keys:
            formation_energy_per_atom
            band_gap
            model_id
            data_sources_used

        Optional added keys:
            confidence_level
            confidence_score
            discrepancy_available
            matched_database_entry
            source_values
            source_discrepancies
            discrepancy_handling
            band_gap_treatment
        """
        if not structures:
            return []

        x_values = self._featurize(structures)

        if self.model_ef is None or self.model_bg is None:
            ef_preds = np.full(len(structures), -1.0, dtype=float)
            bg_preds = np.full(len(structures), 1.0, dtype=float)

        else:
            ef_preds = self.model_ef.predict(x_values)
            bg_preds = self.model_bg.predict(x_values)

        predictions = []

        for i, (ef_value, bg_value) in enumerate(zip(ef_preds, bg_preds)):
            discrepancy_info = self._lookup_discrepancy_info(structures[i])

            predictions.append(
                {
                    "formation_energy_per_atom": float(ef_value),
                    "band_gap": max(0.0, float(bg_value)),
                    "model_id": self.model_id,
                    "data_sources_used": self.data_sources_used,
                    "confidence_level": discrepancy_info["confidence_level"],
                    "confidence_score": discrepancy_info["confidence_score"],
                    "discrepancy_available": discrepancy_info["discrepancy_available"],
                    "matched_database_entry": discrepancy_info["matched_database_entry"],
                    "source_values": discrepancy_info["source_values"],
                    "source_discrepancies": discrepancy_info["source_discrepancies"],
                    "discrepancy_handling": discrepancy_info["discrepancy_handling"],
                    "band_gap_treatment": (
                        "Band gap targets were cleaned to require numeric non-negative values. "
                        "No universal DFT-method correction was applied to all entries. "
                        "For matched MP/JARVIS-DFT entries, band-gap disagreement was used "
                        "as a confidence signal during training and reporting."
                    ),
                }
            )

        return predictions

    def describe(self) -> Dict:
        return {
            "team_name": "ProphX",
            "model_type": (
                "Physics-guided descriptor model using MAGPIE, stoichiometry, "
                "oxidation-state, ionic, electronegativity, and structure-derived features"
            ),
            "api_version": "MatFed API v1",
            "model_id": self.model_id,
            "data_sources": self.data_sources_used,
            "requires_pretrained_weights": True,
            "targets": [
                "formation_energy_per_atom",
                "band_gap",
            ],
            "baseline_extended": "RandomForestRegressor + MAGPIE",
            "extra_features": [
                "Stoichiometry",
                "OxidationStates",
                "IonProperty",
                "average electronegativity",
                "minimum electronegativity difference",
                "average absolute oxidation state",
                "volume",
                "volume per atom",
                "inverse-radius proxy from volume per atom",
                "density",
                "lattice parameters",
                "material class flags",
            ],
            "discrepancy_handling": (
                "For materials matched between EuMINe/MP and JARVIS-DFT, "
                "source-specific values were preserved. Absolute MP-JARVIS differences "
                "were used as sample-confidence weights during training. If a prediction-time "
                "structure matches a known entry, source values, discrepancies, and confidence "
                "are reported. Otherwise, the model reports that discrepancy handling was used "
                "during training but cannot be recomputed for that structure."
            ),
            "band_gap_treatment": (
                "Band gap targets were cleaned to require numeric non-negative values. "
                "No universal DFT-method correction was applied to all entries because "
                "DFT method metadata may not be available for every prediction-time input. "
                "For matched MP/JARVIS-DFT entries, band-gap disagreement was used as a "
                "confidence signal."
            ),
            "notes": (
                "Band gap itself is not used as an input feature to avoid target leakage. "
                "The band-gap model uses physics-guided descriptors inspired by "
                "band-gap correction literature, but no universal correction is applied "
                "unless source-specific information is available."
            ),
        }


# Compatibility aliases.
# These make the class usable under different evaluator naming conventions.
PhysicsGuidedPredictor = ProphXPredictor
Predictor = ProphXPredictor
RandomForestPredictor = ProphXPredictor
