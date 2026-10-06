"""Geometry holon: from a CT (and a lumen segmentation plus a few landmarks) to an
``Anatomy`` with declared uncertainty.

Creaon (what it accepts)
    * a CT volume in Hounsfield units (NIfTI or DICOM series),
    * a lumen mask of the aortic root / LVOT (e.g. TotalSegmentator ``aorta`` ∪
      ``heart_ventricle_left``), same grid as the CT or resampled to it,
    * landmarks placed by the user (3D Slicer markups or a plain JSON): the three cusp
      nadirs that define the annulus plane, the two coronary ostia, and the membranous
      septum length measured in the coronal oblique view.

Genon (what it emits)
    * annulus area, perimeter and area-derived diameter on the plane through the nadirs,
    * upper-LVOT calcium volume (HU threshold inside a band below the annulus plane),
    * coronary ostia heights above the annulus plane,
    * membranous-septum length (manual, with its declared uncertainty),
    each as an :class:`~tavr_decide.anatomy.Uncertain`.

The works are ordinary image geometry (plane fit, plane-mask intersection, thresholding).
What the human still supervises is explicit: the nadirs, the ostia and the membranous
septum are landmarks, because no open model segments them reliably; this is the
component the coded half cannot yet delegate, and the report says so.

Uncertainty defaults are PLACEHOLDERS (see ``UNCERTAINTY_DEFAULTS``) to be replaced by
inter-observer measurements on a real cohort.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .anatomy import Anatomy, Uncertain

# --- declared defaults ---------------------------------------------------------------------
UNCERTAINTY_DEFAULTS = {
    "annulus_diameter_sd_mm": 0.6,      # PLACEHOLDER: inter-observer sd of area-derived diameter
    "calcium_rel_sd": 0.25,             # PLACEHOLDER: relative sd of calcium volume (threshold + partial volume)
    "coronary_height_sd_mm": 1.0,       # PLACEHOLDER
    "ms_length_sd_mm": 1.0,             # PLACEHOLDER: manual measurement sd
}
CALCIUM_HU_THRESHOLD = 850.0            # common practice for contrast-enhanced CT; cohort-specific
# "Upper LVOT" band below the annulus plane, in mm (negative = ventricular side). The
# published cut-off (>= 21 mm^3, AUC 0.80) refers to the upper LVOT; the exact band used in
# that study must be confirmed before comparing numbers. PLACEHOLDER.
UPPER_LVOT_BAND_MM = (-3.0, 0.0)
WALL_DILATION_MM = 2.0                  # calcium sits in the wall just outside the lumen mask


# --- landmarks -----------------------------------------------------------------------------
@dataclass
class Landmarks:
    """World coordinates in mm, RAS (NIfTI convention). Slicer markups in LPS are converted."""
    nadirs: list[tuple[float, float, float]]            # >= 3 points: hinge points / cusp nadirs
    lcc_ostium: tuple[float, float, float] | None = None
    rcc_ostium: tuple[float, float, float] | None = None
    ms_length_mm: float | None = None
    ms_length_sd_mm: float = UNCERTAINTY_DEFAULTS["ms_length_sd_mm"]
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "Landmarks":
        """Accept either this module's plain JSON or a 3D Slicer markups file (.mrk.json).

        Plain JSON:
            {"nadirs": [[x,y,z],...], "lcc_ostium": [x,y,z], "rcc_ostium": [x,y,z],
             "ms_length_mm": 4.2, "ms_length_sd_mm": 1.0}
        Slicer markups: control points labelled 'nadir*', 'LCC', 'RCC' (case-insensitive);
        the membranous-septum length is read from a line markup labelled 'MS' if present.
        """
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if "markups" in d:
            return cls._from_slicer(d)
        return cls(
            nadirs=[tuple(map(float, p)) for p in d["nadirs"]],
            lcc_ostium=tuple(map(float, d["lcc_ostium"])) if d.get("lcc_ostium") else None,
            rcc_ostium=tuple(map(float, d["rcc_ostium"])) if d.get("rcc_ostium") else None,
            ms_length_mm=d.get("ms_length_mm"),
            ms_length_sd_mm=float(d.get("ms_length_sd_mm", UNCERTAINTY_DEFAULTS["ms_length_sd_mm"])),
            extra={k: v for k, v in d.items() if k not in {"nadirs", "lcc_ostium", "rcc_ostium", "ms_length_mm", "ms_length_sd_mm"}},
        )

    @classmethod
    def _from_slicer(cls, d: dict) -> "Landmarks":
        nadirs, lcc, rcc, ms = [], None, None, None
        for m in d["markups"]:
            cs = (m.get("coordinateSystem") or "LPS").upper()
            def to_ras(p):
                x, y, z = map(float, p)
                return (-x, -y, z) if cs == "LPS" else (x, y, z)
            pts = [(cp.get("label", ""), to_ras(cp["position"])) for cp in m.get("controlPoints", [])]
            if m.get("type") == "Line" and any("ms" in (lbl or "").lower() for lbl, _ in pts) and len(pts) == 2:
                ms = float(np.linalg.norm(np.subtract(pts[0][1], pts[1][1])))
                continue
            for lbl, p in pts:
                l = (lbl or "").lower()
                if l.startswith("nadir") or l.startswith("hinge"):
                    nadirs.append(p)
                elif l in ("lcc", "lm", "left"):
                    lcc = p
                elif l in ("rcc", "rca", "right"):
                    rcc = p
        if len(nadirs) < 3:
            raise ValueError("Slicer markups must contain at least three control points labelled 'nadir*'")
        return cls(nadirs=nadirs, lcc_ostium=lcc, rcc_ostium=rcc, ms_length_mm=ms)


# --- plane geometry ------------------------------------------------------------------------
@dataclass(frozen=True)
class Plane:
    center: np.ndarray      # (3,) mm
    normal: np.ndarray      # (3,) unit, pointing from the ventricle toward the aorta
    u: np.ndarray           # in-plane basis
    v: np.ndarray

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=float) - self.center) @ self.normal

    def to_world(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.center + np.outer(a, self.u) + np.outer(b, self.v)

    def with_x_toward(self, point_world: np.ndarray) -> "Plane":
        """Same plane with the in-plane x axis pointing toward ``point_world`` (projected).
        Gives angles an anatomical meaning, e.g. 0 degrees at the non-coronary cusp nadir."""
        d = np.asarray(point_world, dtype=float) - self.center
        d = d - (d @ self.normal) * self.normal
        n = np.linalg.norm(d)
        if n < 1e-9:
            return self
        u = d / n
        return Plane(self.center, self.normal, u, np.cross(self.normal, u))


def fit_plane(points: np.ndarray, toward: np.ndarray | None = None) -> Plane:
    """Least-squares plane through >= 3 points; normal oriented toward ``toward`` if given."""
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 3:
        raise ValueError("need at least three points to fit the annulus plane")
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[-1]
    if toward is not None and (np.asarray(toward) - c) @ n < 0:
        n = -n
    n = n / np.linalg.norm(n)
    u = vt[0] - (vt[0] @ n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return Plane(c, n, u, v)


# --- image access --------------------------------------------------------------------------
@dataclass
class Volume:
    data: np.ndarray        # (i, j, k)
    affine: np.ndarray      # 4x4 voxel -> world (RAS, mm)

    @property
    def spacing(self) -> np.ndarray:
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def voxel_volume_mm3(self) -> float:
        return float(abs(np.linalg.det(self.affine[:3, :3])))

    def world_to_voxel(self, pts: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(self.affine)
        P = np.c_[np.asarray(pts, dtype=float), np.ones(len(pts))]
        return (P @ inv.T)[:, :3]

    def voxel_centers_world(self, idx: np.ndarray) -> np.ndarray:
        P = np.c_[idx.astype(float), np.ones(len(idx))]
        return (P @ self.affine.T)[:, :3]

    def sample_nearest(self, pts_world: np.ndarray, fill=0):
        ijk = np.rint(self.world_to_voxel(pts_world)).astype(int)
        inside = np.all((ijk >= 0) & (ijk < np.array(self.data.shape)), axis=1)
        out = np.full(len(ijk), fill, dtype=self.data.dtype)
        sel = ijk[inside]
        out[inside] = self.data[sel[:, 0], sel[:, 1], sel[:, 2]]
        return out

    def sample_linear(self, pts_world: np.ndarray, fill=0.0) -> np.ndarray:
        """Trilinear interpolation (sub-voxel level sets of a mask become smooth)."""
        from scipy import ndimage
        ijk = self.world_to_voxel(pts_world).T
        return ndimage.map_coordinates(self.data.astype(np.float32), ijk, order=1, mode="constant", cval=fill)


def load_volume(path: str | Path) -> Volume:
    """NIfTI (.nii/.nii.gz) via nibabel, or a DICOM directory via SimpleITK (converted to RAS)."""
    p = Path(path)
    if p.is_dir():
        import SimpleITK as sitk
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(p))
        if not files:
            raise ValueError(f"no DICOM series found in {p}")
        reader.SetFileNames(files)
        img = reader.Execute()
        arr = sitk.GetArrayFromImage(img)           # (k, j, i)
        data = np.transpose(arr, (2, 1, 0)).astype(np.float32)
        sp = np.array(img.GetSpacing()); o = np.array(img.GetOrigin()); d = np.array(img.GetDirection()).reshape(3, 3)
        aff_lps = np.eye(4); aff_lps[:3, :3] = d * sp; aff_lps[:3, 3] = o
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        return Volume(data, lps_to_ras @ aff_lps)
    import nibabel as nib
    img = nib.load(str(p))
    return Volume(np.asarray(img.dataobj, dtype=np.float32), np.asarray(img.affine, dtype=float))


# --- measurements --------------------------------------------------------------------------
def annulus_section(lumen: Volume, plane: Plane, half_width_mm: float = 25.0, step_mm: float = 0.25) -> dict:
    """Intersect the lumen mask with the annulus plane; return area, perimeter and the
    area-derived diameter (the quantity used for sizing), plus the 2D section for QC."""
    from skimage import measure
    a = np.arange(-half_width_mm, half_width_mm + step_mm, step_mm)
    A, B = np.meshgrid(a, a, indexing="ij")
    pts = plane.to_world(A.ravel(), B.ravel())
    field = lumen.sample_linear(pts, fill=0.0).reshape(A.shape)   # smooth 0..1 occupancy
    inside = field >= 0.5
    # keep the connected component that contains the plane centre
    lab = measure.label(inside, connectivity=1)
    c = lab[A.shape[0] // 2, A.shape[1] // 2]
    if c == 0:
        nz = np.argwhere(lab > 0)
        if len(nz) == 0:
            raise ValueError("annulus plane does not intersect the lumen mask")
        d2 = ((nz - np.array(A.shape) / 2) ** 2).sum(axis=1)
        c = lab[tuple(nz[np.argmin(d2)])]
    sec = lab == c
    area = float(sec.sum()) * step_mm ** 2
    # Perimeter from the sub-voxel 0.5 level set of the interpolated occupancy, restricted to
    # the selected component (a staircase contour of the binary mask would overestimate a
    # circle by ~4/pi).
    from scipy import ndimage
    sigma_px = float(lumen.spacing.min() / step_mm)      # one voxel: removes voxel-scale ripple
    smooth = ndimage.gaussian_filter(np.where(sec, field, 0.0), sigma_px)
    perim = 0.0
    for cont in measure.find_contours(smooth, 0.5):
        perim = max(perim, float(np.sum(np.linalg.norm(np.diff(cont, axis=0), axis=1))) * step_mm)
    return {"area_mm2": area, "perimeter_mm": perim,
            "diameter_area_mm": float(2.0 * np.sqrt(area / np.pi)),
            "diameter_perimeter_mm": float(perim / np.pi), "section": sec, "step_mm": step_mm}


def calcium_volume_mm3(ct: Volume, lumen: Volume, plane: Plane, band_mm=UPPER_LVOT_BAND_MM,
                       hu_threshold: float = CALCIUM_HU_THRESHOLD, wall_dilation_mm: float = WALL_DILATION_MM) -> dict:
    """Calcium (HU >= threshold) within a signed-distance band from the annulus plane, inside the
    lumen mask dilated by ``wall_dilation_mm`` so that wall calcium is included."""
    from scipy import ndimage
    if ct.data.shape != lumen.data.shape:
        raise ValueError("CT and lumen mask must share the same grid (resample first)")
    it = int(np.ceil(wall_dilation_mm / lumen.spacing.min()))
    region = ndimage.binary_dilation(lumen.data > 0, iterations=max(it, 1))
    idx = np.argwhere(region)
    d = plane.signed_distance(lumen.voxel_centers_world(idx))
    in_band = (d >= band_mm[0]) & (d <= band_mm[1])
    sel = idx[in_band]
    hu = ct.data[sel[:, 0], sel[:, 1], sel[:, 2]]
    n_ca = int((hu >= hu_threshold).sum())
    return {"volume_mm3": n_ca * ct.voxel_volume_mm3, "n_voxels": n_ca, "band_mm": tuple(band_mm),
            "hu_threshold": hu_threshold}


def coronary_heights(plane: Plane, lm: Landmarks) -> dict:
    out = {}
    if lm.lcc_ostium is not None:
        out["lcc_height_mm"] = float(plane.signed_distance(np.array([lm.lcc_ostium]))[0])
    if lm.rcc_ostium is not None:
        out["rcc_height_mm"] = float(plane.signed_distance(np.array([lm.rcc_ostium]))[0])
    return out


# --- the holon's genon -----------------------------------------------------------------------
def build_anatomy(ct: Volume, lumen: Volume, lm: Landmarks, label: str = "patient",
                  uncertainty: dict | None = None, aorta_hint_world: np.ndarray | None = None) -> tuple[Anatomy, dict]:
    """Return ``(Anatomy, provenance)``. ``provenance`` carries every intermediate number,
    the placeholders used and what the human supplied, for the report."""
    U = dict(UNCERTAINTY_DEFAULTS); U.update(uncertainty or {})
    toward = aorta_hint_world
    if toward is None and lm.extra.get("aorta_point") is not None:
        toward = np.asarray(lm.extra["aorta_point"], dtype=float)
    if toward is None:
        # Default orientation: the aorta is superior to the annulus (+z in RAS) in a
        # standard cardiac CT. Pass ``aorta_hint_world`` or an 'aorta_point' landmark for
        # unusual acquisitions.
        toward = np.mean(np.array(lm.nadirs), axis=0) + np.array([0.0, 0.0, 10.0])
    plane = fit_plane(np.array(lm.nadirs), toward=toward)
    sec = annulus_section(lumen, plane)
    ca = calcium_volume_mm3(ct, lumen, plane)
    ch = coronary_heights(plane, lm)
    if lm.ms_length_mm is None:
        raise ValueError("membranous-septum length is a required manual landmark (ms_length_mm)")
    anat = Anatomy(
        annulus_diameter_mm=Uncertain(sec["diameter_area_mm"], U["annulus_diameter_sd_mm"]),
        ms_length_mm=Uncertain(float(lm.ms_length_mm), float(lm.ms_length_sd_mm)),
        upper_lvot_calcium_mm3=Uncertain(ca["volume_mm3"], max(U["calcium_rel_sd"] * ca["volume_mm3"], 1.0)),
        lcc_coronary_height_mm=Uncertain(ch.get("lcc_height_mm", 14.0), U["coronary_height_sd_mm"]),
        rcc_coronary_height_mm=Uncertain(ch.get("rcc_height_mm", 15.0), U["coronary_height_sd_mm"]),
        label=label,
    )
    prov = {
        "plane_center_ras": plane.center.tolist(), "plane_normal_ras": plane.normal.tolist(),
        "annulus": {k: v for k, v in sec.items() if k != "section"},
        "calcium": ca, "coronary": ch,
        "manual_landmarks": {"nadirs": lm.nadirs, "lcc_ostium": lm.lcc_ostium, "rcc_ostium": lm.rcc_ostium,
                             "ms_length_mm": lm.ms_length_mm},
        "uncertainty_used": U,
        "placeholders": ["UNCERTAINTY_DEFAULTS", "CALCIUM_HU_THRESHOLD", "UPPER_LVOT_BAND_MM", "WALL_DILATION_MM"],
    }
    return anat, prov


# --- optional: TotalSegmentator wrapper ------------------------------------------------------
def run_totalsegmentator(ct_path: str | Path, out_dir: str | Path, fast: bool = True,
                         roi_subset: tuple[str, ...] = ("aorta", "heart_ventricle_left")) -> dict[str, Path]:
    """Call the TotalSegmentator CLI (``total`` task) and return paths of the requested labels.
    Requires ``pip install totalsegmentator``; runs on CPU with ``fast=True`` in minutes."""
    import shutil
    import subprocess
    exe = shutil.which("TotalSegmentator")
    if exe is None:
        raise RuntimeError("TotalSegmentator is not installed (pip install totalsegmentator)")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-i", str(ct_path), "-o", str(out), "--roi_subset", *roi_subset]
    if fast:
        cmd.append("--fast")
    subprocess.run(cmd, check=True)
    return {r: out / f"{r}.nii.gz" for r in roi_subset}


def union_masks(paths: list[str | Path]) -> Volume:
    vols = [load_volume(p) for p in paths]
    data = np.zeros_like(vols[0].data)
    for v in vols:
        if v.data.shape != data.shape:
            raise ValueError("masks must share a grid")
        data = np.maximum(data, (v.data > 0).astype(np.float32))
    return Volume(data, vols[0].affine)
