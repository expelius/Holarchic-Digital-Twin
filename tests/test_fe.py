"""Mesh validity and FEBio input structure. These tests do not need the solver."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as cg

from tavr_decide.frame import FrameSpec, self_expanding_frame
from tavr_decide.fe.lattice import lattice_hex_mesh, cylinder_hex_mesh, hex_signed_volumes
from tavr_decide.fe.febio import DeploymentParams, assemble, write_deployment, read_node_positions

RING = FrameSpec("ring", "self-expanding", 12, 2, 10.0, [(0, 13.0), (1, 13.0)], 0.30, 0.25)


def n_components(mesh) -> int:
    E = mesh.elems
    rows, cols = np.repeat(E, 8, axis=1).ravel(), np.tile(E, (1, 8)).ravel()
    g = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(mesh.nodes),) * 2)
    _, lab = cg.connected_components(g, directed=False)
    return len(np.unique(lab[np.unique(E)]))


@pytest.mark.parametrize("spec", [RING, self_expanding_frame(26, n_cells_circ=12, n_rows=5).spec])
def test_lattice_mesh_is_valid_connected_and_oriented(spec):
    m = lattice_hex_mesh(spec, n_along=4, n_thick=1)
    assert hex_signed_volumes(m.nodes, m.elems).min() > 0, "all hexes must have positive volume"
    assert n_components(m) == 1, "struts must share nodes with their junctions"
    assert len(np.unique(m.elems)) == len(m.nodes), "no orphan nodes"
    for faces, sign in ((m.outer_faces, 1), (m.inner_faces, -1)):
        p = m.nodes[faces]
        nrm = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0]); c = p.mean(1)
        s = np.einsum("ij,ij->i", nrm, np.c_[c[:, 0], c[:, 1], 0 * c[:, 0]])
        assert (sign * s > 0).all()
    r = np.linalg.norm(m.nodes[:, :2], axis=1)
    t = spec.strut_thickness_mm
    rmin, rmax = min(p[1] for p in spec.profile), max(p[1] for p in spec.profile)
    # nodes sample the profile at lattice heights, so they stay within its bounds (± half thickness)
    assert r.min() >= rmin - t / 2 - 1e-6 and r.max() <= rmax + t / 2 + 1e-6


def test_strut_width_too_large_is_rejected():
    bad = FrameSpec("bad", "self-expanding", 40, 2, 10.0, [(0, 5.0), (1, 5.0)], 0.6, 0.25)
    with pytest.raises(ValueError):
        lattice_hex_mesh(bad)


def test_cylinder_mesh_normals_and_volume():
    c = cylinder_hex_mesh(11.0, 2.0, 0.0, 10.0, n_theta=48, n_z=4, n_r=1)
    assert hex_signed_volumes(c.nodes, c.elems).min() > 0
    p = c.nodes[c.inner_faces]; n = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0]); ctr = p.mean(1)
    assert (np.einsum("ij,ij->i", n, np.c_[ctr[:, 0], ctr[:, 1], 0 * ctr[:, 0]]) < 0).all()   # into the lumen


def test_deployment_input_is_well_formed(tmp_path):
    frame = lattice_hex_mesh(RING, n_along=3, n_thick=1)
    vessel = cylinder_hex_mesh(11.0, 2.0, -3.0, 13.0, n_theta=24, n_z=4)
    sleeve = cylinder_hex_mesh(13.45, 0.5, -1.0, 11.0, n_theta=24, n_z=3)
    A = assemble(frame, vessel, sleeve)
    assert all(len(A.node_sets[k]) >= 1 for k in ("frame_a0", "frame_a90", "frame_a180"))
    f = write_deployment(tmp_path / "d.feb", A, DeploymentParams(sleeve_r0=13.45))
    root = ET.parse(f).getroot()
    assert root.attrib["version"] == "4.0"
    assert len(root.find("Mesh/Nodes")) == len(A.nodes)
    assert [e.attrib["name"] for e in root.findall("Mesh/Elements")] == ["frame", "vessel", "sleeve"]
    assert {s.attrib["name"] for s in root.findall("Mesh/Surface")} == {"frame_outer", "vessel_inner", "sleeve_inner"}
    steps = root.findall("Step/step")
    assert [s.attrib["name"] for s in steps] == ["crimp", "release"]
    assert steps[1].find("Contact/contact").attrib["surface_pair"] == "frame_vessel"
    assert root.find("Contact/contact").attrib["surface_pair"] == "frame_sleeve"
    # the sleeve's radial map has one value per sleeve node and unit norm
    ux = [float(n.text) for n in root.find("MeshData/NodeData[@name='sleeve_ux']")]
    uy = [float(n.text) for n in root.find("MeshData/NodeData[@name='sleeve_uy']")]
    # the sleeve map is the node position, i.e. uniform in-plane scaling about the axis
    assert len(ux) == len(A.node_sets["sleeve_all"]) and np.hypot(ux, uy).min() == pytest.approx(13.45, abs=1e-3)
    assert "min_residual" in f.read_text(encoding="ISO-8859-1")


def test_frame_without_axis_aligned_junctions_is_rejected():
    spec = FrameSpec("odd", "self-expanding", 15, 2, 10.0, [(0, 13.0), (1, 13.0)], 0.30, 0.25)
    frame = lattice_hex_mesh(spec, n_along=2)
    with pytest.raises(ValueError):
        assemble(frame, cylinder_hex_mesh(11, 2, -3, 13, 24, 2), cylinder_hex_mesh(13.45, 0.5, -1, 11, 24, 2))


def test_node_log_parser(tmp_path):
    p = tmp_path / "n.txt"
    p.write_text("*Step  = 1\n*Time  = 0.5\n*Data  = x;y;z\n1 1.0 0.0 0.0\n2 0.0 1.0 0.0\n"
                 "*Step  = 2\n*Time  = 1\n*Data  = x;y;z\n1 2.0 0.0 0.0\n2 0.0 2.0 0.0\n", encoding="latin-1")
    d = read_node_positions(p)
    assert sorted(d) == [0.5, 1.0] and d[1.0].shape == (2, 4) and d[1.0][0, 1] == 2.0


# --- integration: needs the FEBio binary (WSL on Windows, or febio4 on PATH) -----------------
def _febio_available() -> bool:
    import os, shutil, subprocess
    from tavr_decide.fe.febio import FEBIO_WSL
    try:
        if os.name == "nt":
            r = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "-u", "root", "--", "test", "-x", FEBIO_WSL],
                               capture_output=True, timeout=60)
            return r.returncode == 0
        return shutil.which(os.environ.get("FEBIO", "febio4")) is not None
    except Exception:
        return False


@pytest.mark.skipif(not _febio_available(), reason="FEBio binary not available")
def test_ring_deployment_reaches_equilibrium_between_vessel_and_free_radius(tmp_path):
    from tavr_decide.fe.deploy import run_deployment
    spec = FrameSpec("ring", "self-expanding", 8, 2, 10.0, [(0, 13.0), (1, 13.0)], 0.30, 0.25)
    res = run_deployment(spec, vessel_radius_mm=11.0, workdir=tmp_path, vessel_z=(-3.0, 13.0), n_along=2,
                         n_theta=48, n_z_vessel=6, n_r_vessel=2, n_bands=3, timeout_s=900)
    assert res.normal_termination, res.summary()
    assert res.times[-1] == pytest.approx(2.0)
    r = res.band_radius_deployed
    assert np.all(r > 11.0) and np.all(r < 13.0), "deployed frame must sit between the vessel and its free radius"
    assert res.equilibrium_drift_mm < 0.01, "the frame must be at rest once the sleeve has left"
    crimped = res.mean_radius_t[np.argmin(np.abs(res.times - 1.0))]
    assert crimped < 11.0, "the crimped frame must clear the vessel before release"


# --- post-processing on a synthetic deployed state (no solver) ---------------------------------
def _synthetic_result(wall_bulge_mm=0.0, push_mm=0.0):
    from tavr_decide.fe.deploy import DeploymentResult
    from tavr_decide.fe.lattice import tube_hex_mesh
    spec = FrameSpec("ring", "self-expanding", 12, 2, 6.0, [(0, 11.0), (1, 11.0)], 0.30, 0.25)
    frame = lattice_hex_mesh(spec, n_along=2)
    frame.nodes[:, 2] -= 3.0                                   # frame spans z in [-3, 3]
    nt = 48
    R = np.full((9, nt), 11.125)                               # wall touching the frame's outer surface
    vessel = tube_hex_mesh(R, np.zeros((9, 2)), np.linspace(-8, 8, 9), 2.0, 1)
    vfinal = vessel.nodes.copy()
    th = np.arctan2(vfinal[:, 1], vfinal[:, 0]); r = np.linalg.norm(vfinal[:, :2], axis=1)
    bulge = wall_bulge_mm * (np.abs(th) < np.deg2rad(20))      # a 40-degree sector where the wall stands off
    rr = r + bulge + push_mm * (vfinal[:, 2] <= -4.0)          # uniform outward push below z = -4
    vfinal[:, 0], vfinal[:, 1] = rr * np.cos(th), rr * np.sin(th)
    return DeploymentResult(True, 1.0, 0, 0, np.array([2.0]), np.array([11.0]), np.zeros(1), np.ones(1), np.ones(1),
                            0.0, None, {}, frame, vessel, frame.nodes.copy(), vfinal, None)


def test_sealing_gap_is_zero_when_conforming_and_matches_a_known_sector():
    from tavr_decide.fe.post import sealing_gap
    assert sealing_gap(_synthetic_result(), z_band=(-3, 1), tol_mm=0.05)["area_mm2"] == pytest.approx(0.0, abs=1e-6)
    g = sealing_gap(_synthetic_result(wall_bulge_mm=1.0), z_band=(-3, 1), tol_mm=0.0)
    expected = np.deg2rad(40) * 0.5 * ((11.125 + 1.0) ** 2 - 11.125 ** 2)     # annular sector of 40 degrees
    assert g["area_mm2"] == pytest.approx(expected, rel=0.15) and g["max_gap_mm"] == pytest.approx(1.0, abs=0.05)


def test_wall_displacement_reads_the_push_below_the_given_height():
    from tavr_decide.fe.post import wall_displacement
    w = wall_displacement(_synthetic_result(push_mm=0.4), z_max=-4.0)
    assert w["p90_mm"] == pytest.approx(0.4, abs=1e-6) and w["n"] > 0
    assert wall_displacement(_synthetic_result(push_mm=0.4), z_max=-20.0)["n"] == 0
    assert wall_displacement(_synthetic_result(), z_max=-4.0)["max_mm"] == pytest.approx(0.0, abs=1e-9)


def test_patient_vessel_recovers_radius_ellipse_and_calcium_location():
    from tests.test_geometry import phantom, landmarks, Z_ANN, SP, SHAPE
    from tavr_decide.geometry import fit_plane, Volume
    from tavr_decide.fe.vessel import vessel_from_lumen
    ct, lumen, _, (cx, cy) = phantom()
    lm = landmarks(cx, cy)
    plane = fit_plane(np.array(lm.nadirs), toward=np.array([cx, cy, Z_ANN + 20])).with_x_toward(np.array([cx + 50, cy, Z_ANN]))
    pv = vessel_from_lumen(lumen, plane, z_range=(-12, 10), n_theta=48, n_z=11, n_r=2, ct=ct)
    assert pv.inner_radius.mean() == pytest.approx(11.0, abs=0.15)
    assert hex_signed_volumes(pv.mesh.nodes, pv.mesh.elems).min() > 0
    ce = pv.mesh.nodes[pv.mesh.elems[pv.calcified]].mean(1)
    assert len(ce) > 0 and np.abs(np.degrees(np.arctan2(ce[:, 1], ce[:, 0]))).max() < 20 and ce[:, 2].max() < 0
    i, j, _k = np.indices(SHAPE)
    ell = ((((i * SP - cx) / 13.0) ** 2 + ((j * SP - cy) / 10.0) ** 2) <= 1).astype(np.float32)
    pe = vessel_from_lumen(Volume(ell, lumen.affine), plane, z_range=(-6, 6), n_theta=48, n_z=4, n_r=1)
    assert pe.inner_radius[2, 0] == pytest.approx(13.0, abs=0.2) and pe.inner_radius[2, 12] == pytest.approx(10.0, abs=0.2)


def test_fe_links_are_monotone_and_anchored():
    from tavr_decide.fe.rungs import conduction_link, pvl_link
    import tavr_decide.calibration as C
    assert conduction_link(C.FE_COND_W0_MM) == pytest.approx(C.DMSID_RISK_AT_CUTOFF)
    assert conduction_link(1.5) > conduction_link(0.1)
    assert pvl_link(C.FE_PVL_G0_MM2) == pytest.approx(C.PVL_RISK_AT_CUTOFF) and pvl_link(10.0) > pvl_link(0.0)


def test_pvl_risk_from_rvol_is_anchored_at_the_varc3_threshold():
    from tavr_decide.fe.rungs import pvl_risk_from_rvol
    assert pvl_risk_from_rvol(30.0, 0.7) == pytest.approx(0.5)
    assert pvl_risk_from_rvol(1.0, 0.7) < 0.01 and pvl_risk_from_rvol(120.0, 0.7) > 0.95
    assert pvl_risk_from_rvol(0.0, 0.7) == pvl_risk_from_rvol(0.05, 0.7)       # floor, no log(0)
    assert pvl_risk_from_rvol(15.0, 0.4) < pvl_risk_from_rvol(15.0, 0.7)        # tighter model, sharper grading


def test_leak_holon_has_three_rungs_and_the_cfd_rung_reads_its_own_solve():
    from types import SimpleNamespace
    from tavr_decide import Anatomy, Uncertain, evolut_like_grammar
    from tavr_decide.fe.rungs import FEEngine, pvl_risk_from_rvol
    import tavr_decide.calibration as C
    anat = Anatomy(Uncertain(22.0, 0.6), Uncertain(4.0, 1.0), Uncertain(20.0, 5.0), observed_depth_mm=Uncertain(4.0, 0.8))
    eng = FEEngine(anat, vessel=SimpleNamespace(), workdir=".")
    act = evolut_like_grammar(size_in_situ=29, retarget_depths=(3.0,)).actions("assess")[0]
    key = (29, 4.0)
    eng.cache[key] = {"key": key, "normal": True, "wall_s": 200.0, "rvol_0d_ml": 1.1, "wall_p90_mm": 0.2}
    eng.runs.append(eng.cache[key])
    eng.cfd_cache[key] = {"key": key, "normal": True, "wall_s": 1200.0, "rvol_cfd_ml": 40.0}
    eng.cfd_runs.append(eng.cfd_cache[key])
    hol = eng.holons()
    pvl = hol["pvl"]
    assert [r.fidelity for r in pvl.rungs] == ["low", "mid", "high"]
    assert pvl.rungs[0].cost_s < pvl.rungs[1].cost_s < pvl.rungs[2].cost_s
    assert pvl.rungs[2].cost_s == pytest.approx(1400.0)
    assert pvl.rungs[2].error_sd < pvl.rungs[1].error_sd
    nominal = eng._nominal()
    pvl.active = 1; mid = float(pvl.predict(nominal, act)[0])
    pvl.active = 2; high = float(pvl.predict(nominal, act)[0])
    assert mid == pytest.approx(pvl_risk_from_rvol(1.1, C.PVL_0D_SIGMA_LOG), abs=1e-9)
    assert high == pytest.approx(pvl_risk_from_rvol(40.0, C.PVL_CFD_SIGMA_LOG), abs=1e-9)
    assert high > 0.5 > mid
    assert [r.fidelity for r in eng.holons(with_cfd=False)["pvl"].rungs] == ["low", "mid"]
