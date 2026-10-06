import json

import nibabel as nib
import numpy as np

from tavr_decide.cli import main
from tests.test_geometry import phantom, landmarks


def test_from_ct_end_to_end_on_phantom(tmp_path, capsys):
    ct, lumen, _, (cx, cy) = phantom()
    nib.save(nib.Nifti1Image(ct.data, ct.affine), str(tmp_path / "ct.nii.gz"))
    nib.save(nib.Nifti1Image(lumen.data, lumen.affine), str(tmp_path / "lumen.nii.gz"))
    lm = landmarks(cx, cy)
    (tmp_path / "lm.json").write_text(json.dumps({"nadirs": lm.nadirs, "lcc_ostium": lm.lcc_ostium,
                                                   "rcc_ostium": lm.rcc_ostium, "ms_length_mm": 4.2}), encoding="utf-8")
    out = tmp_path / "report.md"
    rc = main(["from-ct", str(tmp_path / "ct.nii.gz"), "--mask", str(tmp_path / "lumen.nii.gz"),
               "--landmarks", str(tmp_path / "lm.json"), "--observed-depth", "4.6", "--n", "800",
               "--out", str(out)])
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert "Recommended action" in md and "Geometry provenance" in md and "placeholders" in md
    assert "state 'assess'" in md


def test_from_ct_position_state_without_observed_depth(tmp_path):
    ct, lumen, _, (cx, cy) = phantom()
    nib.save(nib.Nifti1Image(ct.data, ct.affine), str(tmp_path / "ct.nii.gz"))
    nib.save(nib.Nifti1Image(lumen.data, lumen.affine), str(tmp_path / "lumen.nii.gz"))
    lm = landmarks(cx, cy)
    (tmp_path / "lm.json").write_text(json.dumps({"nadirs": lm.nadirs, "ms_length_mm": 4.2}), encoding="utf-8")
    out = tmp_path / "r.md"
    main(["from-ct", str(tmp_path / "ct.nii.gz"), "--mask", str(tmp_path / "lumen.nii.gz"),
          "--landmarks", str(tmp_path / "lm.json"), "--n", "500", "--out", str(out)])
    md = out.read_text(encoding="utf-8")
    assert "state 'position'" in md and "position/size26" in md
