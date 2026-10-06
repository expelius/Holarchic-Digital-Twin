import xml.etree.ElementTree as ET

import numpy as np
import pytest

from tavr_decide.cfd.channel import GapMap, channel_mesh
from tavr_decide.cfd.lumped import pvl_lumped, grade_of, MMHG, RHO, MU, K_JET
from tavr_decide.fe.lattice import hex_signed_volumes


def uniform_gap(h_mm=0.3, r_mm=12.0, L_mm=10.0, nt=72, nz=21):
    th = (np.arange(nt) + 0.5) * 2 * np.pi / nt
    z = np.linspace(0.0, L_mm, nz)
    return GapMap(th, z, np.full((nz, nt), r_mm), np.full((nz, nt), r_mm + h_mm))


def test_lumped_matches_poiseuille_in_the_viscous_limit():
    h, r, L, dp = 0.3, 12.0, 10.0, 0.05                      # mm, mm, mm, mmHg (tiny: inertia negligible)
    p = pvl_lumped(uniform_gap(h, r, L), dp_mmhg=dp)
    rm = (r + h / 2) / 10
    q_analytic = 2 * np.pi * rm * (h / 10) ** 3 * dp * MMHG / (12 * MU * L / 10)
    assert p.flow_ml_s == pytest.approx(q_analytic, rel=0.01)
    assert p.max_reynolds < 1


def test_lumped_approaches_the_orifice_limit_for_wide_short_gaps():
    h, r, dp = 3.0, 12.0, 60.0
    p = pvl_lumped(uniform_gap(h, r, L_mm=0.5), dp_mmhg=dp)
    A = 2 * np.pi * (r + h / 2) / 10 * h / 10
    q_orifice = A * np.sqrt(2 * dp * MMHG / (K_JET * RHO))
    assert p.flow_ml_s == pytest.approx(q_orifice, rel=0.05)


def test_a_strip_sealed_anywhere_carries_no_flow_and_grades_are_varc3():
    g = uniform_gap(0.5)
    r_wall = g.r_wall.copy(); r_wall[5, :36] = g.r_skirt[5, :36]          # half the circumference sealed at one height
    sealed = GapMap(g.theta, g.z, g.r_skirt, r_wall)
    full, half = pvl_lumped(g), pvl_lumped(sealed)
    assert half.flow_ml_s == pytest.approx(full.flow_ml_s / 2, rel=0.02)
    assert np.all(half.strip_flow_ml_s[:36] == 0)
    assert grade_of(0.5) == "none/trace" and grade_of(10) == "mild" and grade_of(45) == "moderate" and grade_of(75) == "severe"


def test_channel_mesh_is_valid_with_outward_faces():
    g = uniform_gap(0.5, nt=24, nz=6)
    m = channel_mesh(g, n_h=3)
    assert hex_signed_volumes(m.nodes, m.elems).min() > 0
    assert len(m.elems) == 24 * 3 * 5
    centre = m.nodes[m.elems].mean(1)
    for name, F in m.faces.items():
        p = m.nodes[F]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
        assert (np.einsum("ij,ij->i", n, p.mean(1) - centre[m.face_elems[name]]) > 0).all(), name
    assert {k: len(v) for k, v in m.faces.items()} == {"aortic": 72, "ventricular": 72, "skirt": 120, "wall": 120}
    assert m.nodes[:, 2].max() == pytest.approx(g.z[-1] / 10)      # centimetres


def test_svmp_case_files_are_well_formed(tmp_path):
    from tavr_decide.cfd.svmp import write_case, CFDParams
    m = channel_mesh(uniform_gap(0.5, nt=24, nz=4), n_h=2)
    f = write_case(tmp_path, m, CFDParams(n_steps=10))
    root = ET.parse(f).getroot()
    assert {e.attrib["name"] for e in root.findall("Add_mesh/Add_face")} == set(m.faces)
    bcs = {e.attrib["name"]: e.find("Type").text.strip() for e in root.findall("Add_equation/Add_BC")}
    assert bcs == {"aortic": "Neu", "ventricular": "Neu", "skirt": "Dir", "wall": "Dir"}
    vtu = ET.parse(tmp_path / "mesh" / "mesh-complete.mesh.vtu").getroot()
    piece = vtu.find("UnstructuredGrid/Piece")
    assert int(piece.attrib["NumberOfCells"]) == len(m.elems)
    gid = [int(v) for v in piece.find("PointData/DataArray").text.split()]
    assert gid[0] == 1 and gid[-1] == len(m.nodes)
    vtp = ET.parse(tmp_path / "mesh" / "mesh-surfaces" / "aortic.vtp").getroot().find("PolyData/Piece")
    owners = [int(v) for v in vtp.find("CellData/DataArray").text.split()]
    assert len(owners) == len(m.faces["aortic"]) and min(owners) >= 1 and max(owners) <= len(m.elems)
