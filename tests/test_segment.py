import numpy as np
import pytest

from tavr_decide.geometry import Volume
from tavr_decide.segment import auto_nadirs, lumen_from_masks, CUSP_LABELS


def cusp_phantom():
    """Three cusp 'cups' around a vertical root (z up = toward the aorta). Each cusp is a shell
    sector whose lowest point is known: at angle a_c, radius 10 mm, height z = 0."""
    sp = 0.5
    shape = (80, 80, 60)
    i, j, k = np.indices(shape)
    x, y, z = i * sp - 20, j * sp - 20, k * sp - 5          # z from -5 to 25 mm
    r = np.hypot(x, y); ang = np.degrees(np.arctan2(y, x)) % 360
    aff = np.diag([sp, sp, sp, 1.0]); aff[:3, 3] = (-20, -20, -5)
    cusps, truth = {}, {}
    for name, a0 in (("non_coronary_cusp", 90), ("left_coronary_cusp", 210), ("right_coronary_cusp", 330)):
        d = np.abs(((ang - a0 + 180) % 360) - 180)
        bottom = 2.0 * (d / 60.0) ** 2 * 6            # cusp floor rises toward the commissures
        mask = (d < 60) & (r > 6) & (r < 11) & (z >= bottom) & (z <= bottom + 1.0)
        cusps[name] = Volume(mask.astype(np.float32), aff)
        truth[CUSP_LABELS[name]] = np.array([10.0 * np.cos(np.radians(a0)) * 0.85, 10.0 * np.sin(np.radians(a0)) * 0.85, 0.25])
    return cusps, truth


def test_auto_nadirs_find_the_lowest_point_of_each_cusp():
    cusps, truth = cusp_phantom()
    nad = auto_nadirs(cusps, toward_aorta=np.array([0.0, 0.0, 30.0]))
    assert set(nad) == {"nadir_N", "nadir_L", "nadir_R"}
    for k, p in nad.items():
        p = np.array(p)
        assert abs(p[2] - truth[k][2]) < 0.6, (k, p)                              # at the cusp floor
        ang_found = np.degrees(np.arctan2(p[1], p[0])) % 360
        ang_true = np.degrees(np.arctan2(truth[k][1], truth[k][0])) % 360
        assert abs(((ang_found - ang_true + 180) % 360) - 180) < 8, (k, ang_found, ang_true)


def test_lumen_union_ignores_non_lumen_classes(tmp_path):
    import nibabel as nib
    a = np.zeros((10, 10, 10), np.uint8); a[2:4] = 1
    b = np.zeros((10, 10, 10), np.uint8); b[6:8] = 1
    c = np.ones((10, 10, 10), np.uint8)
    for n, arr in (("aorta", a), ("left_ventricular_outflow_tract", b), ("annulus_proper", c)):
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(tmp_path / f"{n}.nii.gz"))
    v = lumen_from_masks({n: tmp_path / f"{n}.nii.gz" for n in ("aorta", "left_ventricular_outflow_tract", "annulus_proper")})
    assert v.data.sum() == a.sum() + b.sum()                                     # annulus ring is not lumen


def test_clean_env_drops_the_host_interpreter_variables(monkeypatch):
    from tavr_decide.segment import clean_env
    monkeypatch.setenv("PYTHONHOME", "C:/Slicer/lib/Python"); monkeypatch.setenv("PYTHONPATH", "C:/Slicer/lib")
    env = clean_env()
    assert "PYTHONHOME" not in env and "PYTHONPATH" not in env and "PATH" in env
