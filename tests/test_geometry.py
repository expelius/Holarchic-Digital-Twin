"""Phantom tests for the geometry holon: a synthetic root (cylinder lumen) with a known
annulus, a calcium nodule of known volume just below it, and ostia at known heights."""
import json

import numpy as np
import pytest

from tavr_decide.geometry import (Volume, Landmarks, fit_plane, annulus_section, calcium_volume_mm3,
                                  coronary_heights, build_anatomy)

SP = 0.5   # mm isotropic
SHAPE = (120, 120, 160)
R = 11.0   # lumen radius -> diameter 22 mm
Z_ANN = 80 * SP   # annulus plane at z = 40 mm (world)


def phantom():
    i, j, k = np.indices(SHAPE)
    x, y, z = i * SP, j * SP, k * SP
    cx, cy = SHAPE[0] * SP / 2, SHAPE[1] * SP / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    lumen = (r <= R).astype(np.float32)
    ct = np.where(lumen > 0, 300.0, 50.0).astype(np.float32)          # contrast lumen, soft tissue wall
    # calcium nodule: wall arc between r in [R, R+1.5], 1.5 mm below the annulus, 30 degrees wide
    ang = np.arctan2(y - cy, x - cx)
    ca = (r > R) & (r <= R + 1.5) & (z >= Z_ANN - 2.5) & (z < Z_ANN - 0.5) & (np.abs(ang) < np.deg2rad(15))
    ct[ca] = 1200.0
    aff = np.diag([SP, SP, SP, 1.0])
    return Volume(ct, aff), Volume(lumen, aff), int(ca.sum()) * SP ** 3, (cx, cy)


def landmarks(cx, cy):
    nad = [(cx + R * np.cos(t), cy + R * np.sin(t), Z_ANN) for t in (0.0, 2.094, 4.189)]
    return Landmarks(nadirs=nad, lcc_ostium=(cx - R, cy, Z_ANN + 13.0), rcc_ostium=(cx, cy + R, Z_ANN + 15.5),
                     ms_length_mm=4.2, ms_length_sd_mm=0.8)


def test_plane_fit_orients_toward_aorta():
    ct, lumen, _, (cx, cy) = phantom()
    lm = landmarks(cx, cy)
    p = fit_plane(np.array(lm.nadirs), toward=np.array([cx, cy, Z_ANN + 20]))
    assert np.allclose(np.abs(p.normal), [0, 0, 1], atol=1e-6) and p.normal[2] > 0
    assert p.center[2] == pytest.approx(Z_ANN)


def test_annulus_section_recovers_diameter_and_perimeter():
    ct, lumen, _, (cx, cy) = phantom()
    lm = landmarks(cx, cy)
    p = fit_plane(np.array(lm.nadirs), toward=np.array([cx, cy, Z_ANN + 20]))
    s = annulus_section(lumen, p)
    assert s["diameter_area_mm"] == pytest.approx(2 * R, abs=0.4)
    assert s["perimeter_mm"] == pytest.approx(2 * np.pi * R, rel=0.04)


def test_calcium_volume_in_upper_lvot_band():
    ct, lumen, true_vol, (cx, cy) = phantom()
    lm = landmarks(cx, cy)
    p = fit_plane(np.array(lm.nadirs), toward=np.array([cx, cy, Z_ANN + 20]))
    ca = calcium_volume_mm3(ct, lumen, p, band_mm=(-3.0, 0.0))
    assert ca["volume_mm3"] == pytest.approx(true_vol, rel=0.05)
    above = calcium_volume_mm3(ct, lumen, p, band_mm=(0.0, 3.0))
    assert above["volume_mm3"] == 0.0


def test_coronary_heights_and_full_anatomy_build(tmp_path):
    ct, lumen, true_vol, (cx, cy) = phantom()
    lm = landmarks(cx, cy)
    anat, prov = build_anatomy(ct, lumen, lm, label="phantom")
    assert anat.annulus_diameter_mm.mean == pytest.approx(2 * R, abs=0.4)
    assert anat.lcc_coronary_height_mm.mean == pytest.approx(13.0, abs=1e-6)
    assert anat.rcc_coronary_height_mm.mean == pytest.approx(15.5, abs=1e-6)
    assert anat.ms_length_mm.mean == 4.2 and anat.ms_length_mm.sd == 0.8
    assert anat.upper_lvot_calcium_mm3.mean == pytest.approx(true_vol, rel=0.05)
    assert "placeholders" in prov and prov["manual_landmarks"]["ms_length_mm"] == 4.2
    # plain-JSON landmarks round trip
    f = tmp_path / "lm.json"
    f.write_text(json.dumps({"nadirs": lm.nadirs, "lcc_ostium": lm.lcc_ostium, "rcc_ostium": lm.rcc_ostium,
                             "ms_length_mm": 4.2}), encoding="utf-8")
    lm2 = Landmarks.from_json(f)
    assert lm2.nadirs == [tuple(p) for p in lm.nadirs] and lm2.ms_length_mm == 4.2


def test_slicer_markups_are_parsed_and_converted_from_lps(tmp_path):
    d = {"markups": [
        {"type": "Fiducial", "coordinateSystem": "LPS", "controlPoints": [
            {"label": "nadir_N", "position": [-10.0, -5.0, 40.0]},
            {"label": "nadir_L", "position": [5.0, -8.7, 40.0]},
            {"label": "nadir_R", "position": [5.0, 8.7, 40.0]},
            {"label": "LCC", "position": [10.0, 0.0, 53.0]},
            {"label": "RCC", "position": [0.0, -10.0, 55.0]}]},
        {"type": "Line", "coordinateSystem": "LPS", "controlPoints": [
            {"label": "MS_1", "position": [0.0, 0.0, 36.0]}, {"label": "MS_2", "position": [0.0, 0.0, 40.2]}]},
    ]}
    f = tmp_path / "m.mrk.json"
    f.write_text(json.dumps(d), encoding="utf-8")
    lm = Landmarks.from_json(f)
    assert lm.nadirs[0] == (10.0, 5.0, 40.0)        # LPS -> RAS flips x and y
    assert lm.lcc_ostium == (-10.0, 0.0, 53.0)
    assert lm.ms_length_mm == pytest.approx(4.2)


def test_missing_ms_length_is_an_error_not_a_default():
    ct, lumen, _, (cx, cy) = phantom()
    lm = landmarks(cx, cy)
    lm.ms_length_mm = None
    with pytest.raises(ValueError):
        build_anatomy(ct, lumen, lm)
