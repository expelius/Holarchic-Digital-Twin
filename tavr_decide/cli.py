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
    md += "\n\n## Geometry provenance\n\n```json\n" + json.dumps(prov, indent=1, default=float) + "\n```\n"
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    return md


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
    p.set_defaults(func=from_ct)
    args = ap.parse_args(argv)
    out = args.func(args)
    if not getattr(args, "out", None):
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
