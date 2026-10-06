"""Rebuild a DeploymentResult from a finished run directory (for runs made before results were pickled)."""
import pickle, numpy as np, nibabel as nib
from tavr_decide.frame import self_expanding_frame
from tavr_decide.fe.lattice import lattice_hex_mesh
from tavr_decide.fe.vessel import vessel_from_lumen
from tavr_decide.fe.deploy import DeploymentResult
from tavr_decide.fe.febio import read_node_positions
from tavr_decide.geometry import Volume

def rebuild(cid, size, depth, wd, along=2):
    case = pickle.load(open(f"data/tavrp_pl/case_ct{cid}.pkl", "rb"))
    img = nib.load(f"data/tavrp_pl/taviHeartData/imagesTs/ct{cid}.nii.gz"); lab = nib.load(f"data/tavrp_pl/taviHeartData/labelsTs/label{cid}.nii.gz")
    ct = Volume(np.asarray(img.dataobj, dtype=np.float32), np.asarray(img.affine, dtype=float)); L = np.asarray(lab.dataobj).astype(np.int16)
    lumen = Volume(np.isin(L, [1, 2, 3, 4, 5]).astype(np.float32), ct.affine)
    pv = vessel_from_lumen(lumen, case["plane"], z_range=(-8, 8), n_theta=48, n_z=8, n_r=2, ct=ct)
    frame = lattice_hex_mesh(self_expanding_frame(size, n_cells_circ=12, n_rows=5).spec, n_along=along, n_thick=1)
    frame.nodes[:, 2] -= depth
    fp = read_node_positions(f"{wd}/frame_nodes.txt"); vp = read_node_positions(f"{wd}/vessel_nodes.txt")
    t = max(fp)
    return DeploymentResult(True, 0.0, 0, 0, np.array([t]), np.zeros(1), np.zeros(1), np.ones(1), np.ones(1), 0.0, wd, {},
                            frame, pv.mesh, fp[t][:, 1:4], vp[max(vp)][:, 1:4], pv.calcified)
