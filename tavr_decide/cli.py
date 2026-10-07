"""Command line: from a CT to a decision report.

    tavr-decide from-ct CT.nii.gz --landmarks lm.mrk.json [--mask lumen.nii.gz] [--segment]
                        [--observed-depth 4.6 --observed-sd 0.8] [--size 26] [--state assess|position]
                        [--out report.md]

Without ``--mask`` and with ``--segment``, TotalSegmentator (task ``total``, CPU ``--fast``)
produces ``aorta`` and ``heart_ventricle_left`` and their union is used as the lumen.
The membranous-septum length must be in the landmarks (manual measurement).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from pathlib import Path

from .anatomy import Uncertain
from .decision import Utility, evaluate, margin_certificate, holarchic_select
from .geometry import Landmarks, build_anatomy, load_volume, run_totalsegmentator, union_masks
from .grammar import evolut_like_grammar
from .modules import ConductionProxy, PVLProxy
from .holon import leaf
from .report import render_markdown

CREAON = ("ms_length_mm", "annulus_diameter_mm", "upper_lvot_calcium_mm3")


def default_holons():
    return {
        "conduction": leaf("conduction", "conduction", CREAON,
                           [("dMSID proxy", "low", 0.001, ConductionProxy().error_sd, ConductionProxy().predict)]),
        "pvl": leaf("paravalvular leak", "pvl", CREAON,
                    [("upper-LVOT calcium proxy", "low", 0.001, PVLProxy().error_sd, PVLProxy().predict)]),
    }


def from_ct(args: argparse.Namespace) -> str:
    ct = load_volume(args.ct)
    if args.mask:
        lumen = load_volume(args.mask)
    elif args.segment:
        paths = run_totalsegmentator(args.ct, Path(args.workdir or Path(args.ct).parent / "segmentation"))
        lumen = union_masks(list(paths.values()))
    else:
        sys.exit("give --mask lumen.nii.gz or --segment (TotalSegmentator)")
    lm = Landmarks.from_json(args.landmarks)
    label = args.label or Path(args.ct).stem.replace(".nii", "")
    anat, prov = build_anatomy(ct, lumen, lm, label=label)
    if args.observed_depth is not None:
        from dataclasses import replace
        anat = replace(anat, observed_depth_mm=Uncertain(args.observed_depth, args.observed_sd))
    state = args.state or ("assess" if anat.observed_depth_mm is not None else "position")
    if state == "assess" and anat.observed_depth_mm is None:
        sys.exit("state 'assess' needs --observed-depth (depth below the NCC on pre-release cine)")
    grammar = evolut_like_grammar(sizes=tuple(args.sizes), size_in_situ=args.size,
                                  retarget_depths=tuple(args.retarget_depths))
    holons = default_holons()
    utility = Utility(w_conduction=args.w_conduction, w_pvl=args.w_pvl)
    res, trace = holarchic_select(anat, grammar, state, holons, utility, eta=args.eta, n=args.n, seed=0)
    certs = []
    for var, shift, step in (("ms_length_mm", 4.0, 0.25), ("upper_lvot_calcium_mm3", 40.0, 2.0),
                             ("annulus_diameter_mm", 2.0, 0.1)):
        certs.append(margin_certificate(anat, grammar, state, holons, utility, var, shift, step, n=min(args.n, 2000)))
    if anat.observed_depth_mm is not None:
        certs.append(margin_certificate(anat, grammar, state, holons, utility, "observed_depth_mm", 3.0, 0.1, n=min(args.n, 2000)))
    md = render_markdown(res, certificates=certs, trace=trace, title=f"TAVR-Decide report — {label} — state '{state}'")
    if args.lang != "none":
        from .clinical import clinical_summary
        md = md.replace("## Recommendation", clinical_summary(res, anat, certs, trace, args.lang) + "\n## Recommendation", 1)
    md += "\n\n## Geometry provenance\n\n```json\n" + json.dumps(prov, indent=1, default=float) + "\n```\n"
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    return md


def _status(workdir: Path, state: str, **extra) -> None:
    import time
    s = {"state": state, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    tmp = workdir / "status.json.tmp"
    tmp.write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    tmp.replace(workdir / "status.json")                  # atomic: the panel never reads half a file


def physics(args: argparse.Namespace) -> str:
    """Full physics for one case, meant to run as a background job: geometry, patient wall,
    finite-element deployment per action, paravalvular channel, 0D and CFD leak, decision.
    Progress and the result are written to <workdir>/status.json and <workdir>/report.md."""
    import traceback
    from dataclasses import replace
    from .clinical import clinical_summary
    from .decision import evaluate
    from .fe.febio import DeploymentParams
    from .fe.rungs import FEEngine
    from .fe.vessel import vessel_from_lumen
    from .geometry import fit_plane
    wd = Path(args.workdir); wd.mkdir(parents=True, exist_ok=True)
    _status(wd, "running", step="geometry")
    try:
        ct, lumen = load_volume(args.ct), load_volume(args.mask)
        lm = Landmarks.from_json(args.landmarks)
        anat, prov = build_anatomy(ct, lumen, lm, label=args.label or Path(args.ct).stem.replace(".nii", ""))
        if args.observed_depth is not None:
            anat = replace(anat, observed_depth_mm=Uncertain(args.observed_depth, args.observed_sd))
        state = args.state or ("assess" if anat.observed_depth_mm is not None else "position")
        grammar = evolut_like_grammar(sizes=tuple(args.sizes), depths=tuple(args.depths), size_in_situ=args.size,
                                      retarget_depths=tuple(args.retarget_depths))
        center = np.array(prov["plane_center_ras"]); normal = np.array(prov["plane_normal_ras"])
        plane = fit_plane(np.array(lm.nadirs), toward=center + 10 * normal).with_x_toward(np.array(lm.nadirs[0]))
        _status(wd, "running", step="patient wall")
        vessel = vessel_from_lumen(lumen, plane, z_range=(-8.0, 8.0), n_theta=48, n_z=8, n_r=2, ct=ct)
        engine = FEEngine(anat, vessel, wd / "fe",
                          fe_kwargs=dict(n_along=2, crimp_margin_mm=0.5, params=DeploymentParams(steps_per_stage=40),
                                         timeout_s=5400),
                          cfd_kwargs=dict(nproc=args.nproc))
        holons = engine.holons(with_cfd=not args.no_cfd)
        utility = Utility()
        n_actions = len(grammar.actions(state))
        _status(wd, "running", step=f"physics for {n_actions} actions (finite elements"
                + ("" if args.no_cfd else " + CFD") + "); minutes per action", actions=n_actions)
        if args.force:
            for h in holons.values():
                h.active = len(h.rungs) - 1
            res = evaluate(anat, grammar, state, holons, utility, n=args.n, seed=0, eta=args.eta)
            trace = [{"round": "forced", "pea_before": float("nan"), "contracts": {}, "unmet": [],
                      "narratives": {o: h.narrate() for o, h in holons.items()}}]
        else:
            res, trace = holarchic_select(anat, grammar, state, holons, utility, eta=args.eta, n=args.n, seed=0)
        rows = ["## Physics per action", "",
                "| action | deployment | wall displacement below MS (p90) | leak 0D (mL/beat) | leak CFD (mL/beat) | grade |",
                "|---|---|---|---|---|---|"]
        for a in grammar.actions(state):
            key = (int(a.size_mm), round(engine.depth_of(a), 2))
            fe, cf = engine.cache.get(key), engine.cfd_cache.get(key, {})
            if fe is None:
                rows.append(f"| {a.key} | not run (not needed) | | | | |"); continue
            rows.append(f"| {a.key} | {'normal' if fe['normal'] else 'FAILED'} {fe['wall_s']:.0f} s | "
                        f"{fe.get('wall_p90_mm', float('nan')):.2f} mm | {fe.get('rvol_0d_ml', float('nan')):.1f} | "
                        f"{cf.get('rvol_cfd_ml', float('nan')):.1f} | {cf.get('grade_cfd', fe.get('grade_0d', ''))} |")
        md = render_markdown(res, trace=trace, title=f"TAVR Decide physics — {anat.label} — state '{state}'")
        md = md.replace("## Recommendation", clinical_summary(res, anat, [], trace, args.lang) + "\n"
                        + "\n".join(rows) + "\n\n## Recommendation", 1)
        (wd / "report.md").write_text(md, encoding="utf-8")
        (wd / "physics.json").write_text(json.dumps({"fe": engine.runs, "cfd": engine.cfd_runs}, indent=1, default=str),
                                         encoding="utf-8")
        _status(wd, "done", recommended=res.best.key, stability=res.pea, fe_runs=len(engine.runs),
                cfd_runs=len(engine.cfd_runs))
        return md
    except Exception as e:
        _status(wd, "error", message=str(e), traceback=traceback.format_exc())
        raise


def segment(args: argparse.Namespace) -> str:
    """Background job: TotalSegmentator -> lumen.nii.gz (+ landmarks_auto.json when licensed)."""
    import time
    import traceback
    from .segment import (CUSP_LABELS, auto_nadirs, find_totalsegmentator, has_licence, lumen_from_masks,
                          run_tasks, set_licence)
    import nibabel as nib
    wd = Path(args.workdir); wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        exe = args.exe or find_totalsegmentator()
        if not exe:
            raise RuntimeError("TotalSegmentator not found: pip install totalsegmentator, or give --exe")
        if args.licence:
            set_licence(args.licence, exe)
        licensed = has_licence(exe)
        _status(wd, "running", step=f"TotalSegmentator ({'licensed' if licensed else 'free'} mode) on {args.device}",
                licensed=licensed)
        masks = run_tasks(args.ct, wd / "totalseg", exe, licensed, device=args.device,
                          log=lambda m: _status(wd, "running", step=m, licensed=licensed))
        lumen = lumen_from_masks(masks)
        nib.save(nib.Nifti1Image(lumen.data.astype(np.uint8), lumen.affine), str(wd / "lumen.nii.gz"))
        out = {"lumen": str(wd / "lumen.nii.gz"), "masks": {k: str(v) for k, v in masks.items()}, "licensed": licensed}
        cusps = {c: load_volume(masks[c]) for c in CUSP_LABELS if c in masks}
        if len(cusps) == 3:
            aorta_c = None
            if "aorta" in masks:
                a = load_volume(masks["aorta"])
                aorta_c = a.voxel_centers_world(np.argwhere(a.data > 0)).mean(axis=0)
            annulus = load_volume(masks["annulus_proper"]) if "annulus_proper" in masks else None
            nad = auto_nadirs(cusps, toward_aorta=aorta_c, annulus=annulus)
            (wd / "landmarks_auto.json").write_text(json.dumps({"nadirs_labelled": nad}, indent=1), encoding="utf-8")
            out["landmarks_auto"] = str(wd / "landmarks_auto.json")
        note = None if licensed else ("free mode: the aorta only; the LVOT below the annulus is not segmented. "
                                      "Complete it by hand or configure a TotalSegmentator academic licence.")
        _status(wd, "done", elapsed_s=round(time.time() - t0), note=note, **out)
        return json.dumps(out)
    except Exception as e:
        _status(wd, "error", message=str(e), traceback=traceback.format_exc())
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tavr-decide", description="Holarchic digital twin for TAVR: decision layer CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-ct", help="CT + lumen mask (or TotalSegmentator) + landmarks -> decision report")
    p.add_argument("ct"); p.add_argument("--landmarks", required=True)
    p.add_argument("--mask"); p.add_argument("--segment", action="store_true"); p.add_argument("--workdir")
    p.add_argument("--label"); p.add_argument("--out")
    p.add_argument("--observed-depth", type=float); p.add_argument("--observed-sd", type=float, default=0.8)
    p.add_argument("--state", choices=["assess", "position"])
    p.add_argument("--size", type=int, default=26); p.add_argument("--sizes", type=int, nargs="+", default=[26, 29])
    p.add_argument("--retarget-depths", type=float, nargs="+", default=[3.0, 4.0, 5.0])
    p.add_argument("--w-conduction", type=float, default=1.0); p.add_argument("--w-pvl", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.95); p.add_argument("--n", type=int, default=4000)
    p.add_argument("--lang", choices=["es", "en", "none"], default="es", help="language of the clinical summary")
    p.set_defaults(func=from_ct)

    q = sub.add_parser("physics", help="finite-element deployment + paravalvular CFD for one case (background job)")
    q.add_argument("ct"); q.add_argument("--mask", required=True); q.add_argument("--landmarks", required=True)
    q.add_argument("--workdir", required=True); q.add_argument("--label"); q.add_argument("--out")
    q.add_argument("--observed-depth", type=float); q.add_argument("--observed-sd", type=float, default=0.8)
    q.add_argument("--state", choices=["assess", "position"])
    q.add_argument("--size", type=int, default=26); q.add_argument("--sizes", type=int, nargs="+", default=[26, 29])
    q.add_argument("--depths", type=float, nargs="+", default=[3.0, 5.0])
    q.add_argument("--retarget-depths", type=float, nargs="+", default=[3.0, 4.0, 5.0])
    q.add_argument("--eta", type=float, default=0.95); q.add_argument("--n", type=int, default=2000)
    q.add_argument("--force", action="store_true", help="open every rung for every action (review mode)")
    q.add_argument("--no-cfd", action="store_true", help="stop at the finite-element + 0D rung")
    q.add_argument("--nproc", type=int, default=6, help="MPI ranks for the CFD")
    q.add_argument("--lang", choices=["es", "en"], default="es")
    q.set_defaults(func=physics)

    s = sub.add_parser("segment", help="TotalSegmentator lumen (+ automatic nadirs when licensed), background job")
    s.add_argument("ct"); s.add_argument("--workdir", required=True); s.add_argument("--out")
    s.add_argument("--exe", help="path to the TotalSegmentator executable (default: auto-detect)")
    s.add_argument("--licence", help="TotalSegmentator licence number (free for academic use); stored once")
    s.add_argument("--device", default="cpu", choices=["cpu", "gpu", "mps"])
    s.set_defaults(func=segment)
    args = ap.parse_args(argv)
    out = args.func(args)
    if not getattr(args, "out", None):
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
