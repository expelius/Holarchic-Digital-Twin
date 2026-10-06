"""First real CT through the pipeline: TAVRP-PL ct0091 (contrast CT, 1.5 mm isotropic, pseudo-labels).

Landmarks are derived automatically from the 'annulus' pseudo-label (plane by PCA; three
pseudo-nadirs on the lumen contour at 120 degrees). They are an approximation, declared as
such; the membranous-septum length and the coronary heights are not in this dataset and are
entered as uncertain placeholders.
"""
import sys, json
import numpy as np, nibabel as nib
from tavr_decide.geometry import Volume, Landmarks, Plane, fit_plane, build_anatomy, annulus_section
from tavr_decide.anatomy import Uncertain

cid = sys.argv[1] if len(sys.argv) > 1 else "0091"
img = nib.load(f"data/tavrp_pl/taviHeartData/imagesTs/ct{cid}.nii.gz"); lab = nib.load(f"data/tavrp_pl/taviHeartData/labelsTs/label{cid}.nii.gz")
ct = Volume(np.asarray(img.dataobj, dtype=np.float32), np.asarray(img.affine, dtype=float))
L = np.asarray(lab.dataobj).astype(np.int16)
lumen = Volume(np.isin(L, [1, 2, 3, 4, 5]).astype(np.float32), ct.affine)      # aorta+root+valve+LV+annulus

# annulus plane from the annulus label (PCA of its voxel centres), normal toward the aorta label
ann_idx = np.argwhere(L == 5)
ann_w = ct.voxel_centers_world(ann_idx)
aorta_c = ct.voxel_centers_world(np.argwhere(L == 2)).mean(0)   # aortic ROOT centroid, not the whole aorta
lv_c = ct.voxel_centers_world(np.argwhere(L == 4)).mean(0)
p0 = fit_plane(ann_w, toward=aorta_c)
print(f"annulus label: {len(ann_idx)} voxels, centre {p0.center.round(1)}, normal {p0.normal.round(3)}")
print(f"  distance centre->aorta centroid along normal: {(aorta_c - p0.center) @ p0.normal:.1f} mm; ->LV centroid: {(lv_c - p0.center) @ p0.normal:.1f} mm")

# pseudo-nadirs: lumen contour on the plane at 0/120/240 degrees
sec = annulus_section(lumen, p0)
print(f"section on the label plane: area {sec['area_mm2']:.0f} mm2, perimeter {sec['perimeter_mm']:.1f} mm, D(area) {sec['diameter_area_mm']:.1f} mm, D(perim) {sec['diameter_perimeter_mm']:.1f} mm")
from skimage import measure
cont = max(measure.find_contours(sec["section"].astype(float), 0.5), key=len)
step = sec["step_mm"]; n = sec["section"].shape[0]
ab = (cont - n / 2) * step                           # in-plane coords (a, b)
ang = np.arctan2(ab[:, 1], ab[:, 0])
nadirs = []
for target in (0.0, 2 * np.pi / 3, -2 * np.pi / 3):
    k = np.argmin(np.abs((ang - target + np.pi) % (2 * np.pi) - np.pi))
    nadirs.append(tuple(p0.to_world(np.array([ab[k, 0]]), np.array([ab[k, 1]]))[0]))
lm = Landmarks(nadirs=nadirs, ms_length_mm=4.0, ms_length_sd_mm=1.5)      # MS not in this dataset: placeholder
anat, prov = build_anatomy(ct, lumen, lm, label=f"tavrp_ct{cid}", aorta_hint_world=aorta_c)
print(f"anatomy: annulus D {anat.annulus_diameter_mm.mean:.1f} ± {anat.annulus_diameter_mm.sd} mm, perimeter {prov['annulus']['perimeter_mm']:.1f} mm, "
      f"upper-LVOT calcium {anat.upper_lvot_calcium_mm3.mean:.0f} mm3, MS {anat.ms_length_mm.mean} ± {anat.ms_length_mm.sd} (placeholder)")
json.dump({"nadirs": [list(map(float, p)) for p in nadirs], "ms_length_mm": 4.0, "ms_length_sd_mm": 1.5,
           "aorta_point": list(map(float, aorta_c))}, open(f"data/tavrp_pl/landmarks_ct{cid}.json", "w"), indent=1)
print("landmarks written")

# --- patient wall from the lumen, local frame oriented along the annulus normal ---------------
from tavr_decide.fe.vessel import vessel_from_lumen
plane = fit_plane(np.array(nadirs), toward=aorta_c).with_x_toward(np.array(nadirs[0]))
pv = vessel_from_lumen(lumen, plane, z_range=(-10, 8), n_theta=48, n_z=9, n_r=2, ct=ct)
print("wall mesh:", len(pv.mesh.elems), "hexes,", int(pv.calcified.sum()), "calcified")
for iz, z in enumerate(pv.zs):
    r = pv.inner_radius[iz]
    print(f"  z {z:6.1f}  r mean {r.mean():5.1f}  min {r.min():5.1f}  max {r.max():5.1f}   centre offset {np.linalg.norm(pv.centers[iz]):4.1f} mm")
import pickle; pickle.dump({"vessel": pv, "anatomy": anat, "plane": plane}, open(f"data/tavrp_pl/case_ct{cid}.pkl", "wb"))
