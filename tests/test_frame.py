import numpy as np
import pytest

from tavr_decide.frame import self_expanding_frame, balloon_expandable_frame


def test_self_expanding_lattice_topology_and_profile():
    f = self_expanding_frame(26, n_cells_circ=15, n_rows=5)
    assert f.nodes.shape == (15 * 6, 3)
    assert f.segments.shape == (2 * 15 * 5, 2)
    # every node connects to 2 (end rings) or 4 (interior) struts
    deg = np.bincount(f.segments.ravel(), minlength=len(f.nodes))
    assert set(deg[f.row_of_node == 0]) == {2} and set(deg[(f.row_of_node > 0) & (f.row_of_node < 5)]) == {4}
    prof = f.radius_profile(6)
    assert prof[0] == pytest.approx(13.0, abs=0.3)           # inflow radius = size/2
    assert prof.min() < 11.0 and prof[-1] > prof[0]          # waist, flared outflow
    assert f.inflow_z == 0.0 and f.outflow_z == pytest.approx(45.0)


def test_crimp_is_radial_only_and_reversible_in_axis():
    f = self_expanding_frame(29)
    c = f.crimped(radius_mm=3.0)
    assert np.allclose(np.linalg.norm(c.nodes[:, :2], axis=1), 3.0)
    assert np.allclose(c.nodes[:, 2], f.nodes[:, 2])
    assert c.strut_length_total_mm() < f.strut_length_total_mm()


def test_hex_mesh_and_vtk_export(tmp_path):
    f = balloon_expandable_frame(23)
    nodes, elems = f.to_hex_mesh()
    assert elems.shape == (len(f.segments), 8) and nodes.shape == (8 * len(f.segments), 3)
    # cross-section edges have the specified width and thickness
    e = elems[0]
    assert np.linalg.norm(nodes[e[1]] - nodes[e[0]]) == pytest.approx(f.spec.strut_thickness_mm, abs=1e-6)
    assert np.linalg.norm(nodes[e[2]] - nodes[e[1]]) == pytest.approx(f.spec.strut_width_mm, abs=1e-6)
    p = tmp_path / "frame.vtk"
    f.to_vtk_lines(str(p))
    txt = p.read_text(encoding="utf-8")
    assert "POLYDATA" in txt and f"LINES {len(f.segments)}" in txt
