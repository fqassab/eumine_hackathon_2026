# ProphX_v2_scaled

## Overview

ProphX_v2_scaled is a physics-guided descriptor model for:

- formation energy per atom, in eV/atom
- electronic band gap, in eV

The package uses separate trained regressors for the two targets and is compatible with the MatFed predictor interface.

## Included artifacts

```text
models/prophx_v2_scaled/
    physics_model_ef.joblib
    physics_model_bg.joblib
    feature_columns.joblib
    metadata.joblib
    discrepancy_lookup.joblib
    ProphX_v2_metadata.json
    ProphX_v2_scaled_metadata.json