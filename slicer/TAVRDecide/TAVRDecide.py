"""3D Slicer module for the Holarchic Digital Twin for TAVR.

Workflow, top to bottom in the panel:
  1. Inputs      CT volume and the lumen segmentation (aortic root + LVOT).
  2. Landmarks   three annulus nadirs (hinge points), the two coronary ostia, and the
                 membranous-septum length as a line. Buttons create the nodes and start
                 placement; labels are set automatically.
  3. Procedure   planning (choose size and target depth) or at ~80 % deployment of a
                 self-expanding valve (continue at the observed depth vs recapture).
  4. Decide      geometry holon -> anatomy with uncertainty -> delegated decision with the
                 published proxies -> recommendation, action-stability probability, margin
                 certificates in mm, report. The annulus plane is added to the scene.

The finite-element and CFD rungs run from the command line for now (minutes to tens of
minutes each); this panel uses the millisecond rungs. Nothing here is a clinical claim.
"""
from __future__ import annotations

import os
from dataclasses import replace

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleTest, ScriptedLoadableModuleWidget)

LANDMARK_LABELS = ["nadir_N", "nadir_L", "nadir_R", "LCC", "RCC"]


# -----------------------------------------------------------------------------------------
class TAVRDecide(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "TAVR Decide"
        self.parent.categories = ["Cardiac"]
        self.parent.dependencies = []
        self.parent.contributors = ["Juan David Zuluaga-Monroy (Pontificia Universidad Javeriana)"]
        self.parent.helpText = (
            "Holarchic digital twin for TAVR: geometry from CT, reversibility grammar of the "
            "procedure, action-stability probability and margin certificates. "
            "https://github.com/expelius/Holarchic-Digital-Twin")
        self.parent.acknowledgementText = (
            "Proxy coefficients are declared placeholders anchored to published cut-offs. "
            "Research software, not for clinical use.")


# -----------------------------------------------------------------------------------------
class TAVRDecideWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = TAVRDecideLogic()

        # 1. inputs
        box = ctk.ctkCollapsibleButton(); box.text = "1. Inputs"; self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.ctSelector = self._combo(["vtkMRMLScalarVolumeNode"], "CT (contrast, ECG-gated if possible)")
        form.addRow("CT volume:", self.ctSelector)
        self.segSelector = self._combo(["vtkMRMLSegmentationNode"], "Lumen segmentation (all segments are joined)")
        form.addRow("Lumen segmentation:", self.segSelector)

        # 2. landmarks
        box = ctk.ctkCollapsibleButton(); box.text = "2. Landmarks"; self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.pointsSelector = self._combo(["vtkMRMLMarkupsFiducialNode"], "Points: nadir_N, nadir_L, nadir_R, LCC, RCC")
        form.addRow("Annulus and ostia:", self.pointsSelector)
        b = qt.QPushButton("Place nadirs and ostia"); b.toolTip = (
            "Creates a point list and starts placement. Place, in order: non-, left- and right-coronary "
            "cusp nadirs, then the left and right coronary ostia.")
        b.connect("clicked()", self.onPlacePoints); form.addRow(b)
        self.msSelector = self._combo(["vtkMRMLMarkupsLineNode"], "Line along the membranous septum")
        form.addRow("Membranous septum:", self.msSelector)
        b = qt.QPushButton("Measure membranous septum"); b.connect("clicked()", self.onPlaceMS); form.addRow(b)
        self.msSd = qt.QDoubleSpinBox(); self.msSd.setRange(0.1, 5.0); self.msSd.setValue(1.0); self.msSd.setSuffix(" mm")
        form.addRow("MS measurement SD:", self.msSd)

        # 3. procedure state
        box = ctk.ctkCollapsibleButton(); box.text = "3. Procedure"; self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.stateCombo = qt.QComboBox()
        self.stateCombo.addItem("Planning: choose size and target depth", "position")
        self.stateCombo.addItem("At ~80% deployment: continue or recapture", "assess")
        self.stateCombo.connect("currentIndexChanged(int)", self.onStateChanged)
        form.addRow("State:", self.stateCombo)
        self.sizes = qt.QLineEdit("26, 29"); form.addRow("Sizes (mm):", self.sizes)
        self.depths = qt.QLineEdit("3, 5"); form.addRow("Target depths (mm):", self.depths)
        self.sizeInSitu = qt.QSpinBox(); self.sizeInSitu.setRange(20, 34); self.sizeInSitu.setValue(26)
        form.addRow("Valve in situ (mm):", self.sizeInSitu)
        self.obsDepth = qt.QDoubleSpinBox(); self.obsDepth.setRange(-5, 15); self.obsDepth.setValue(4.0); self.obsDepth.setSuffix(" mm")
        form.addRow("Observed depth below NCC:", self.obsDepth)
        self.obsSd = qt.QDoubleSpinBox(); self.obsSd.setRange(0.1, 5); self.obsSd.setValue(0.8); self.obsSd.setSuffix(" mm")
        form.addRow("Observed depth SD:", self.obsSd)
        self.retarget = qt.QLineEdit("3, 4, 5"); form.addRow("Recapture to depths (mm):", self.retarget)
        self.eta = qt.QDoubleSpinBox(); self.eta.setRange(0.5, 0.999); self.eta.setDecimals(3); self.eta.setValue(0.95)
        form.addRow("Stability threshold:", self.eta)
        self.onStateChanged(0)

        # 4. decide
        self.decideButton = qt.QPushButton("Decide"); self.decideButton.setStyleSheet("font-weight: bold; padding: 6px")
        self.decideButton.connect("clicked()", self.onDecide); self.layout.addWidget(self.decideButton)
        self.report = qt.QTextBrowser(); self.report.setMinimumHeight(420)
        self.layout.addWidget(self.report, 1)          # the report takes the remaining height
        b = qt.QPushButton("Save report..."); b.connect("clicked()", self.onSave); self.layout.addWidget(b)
        self._markdown = ""

    def _combo(self, types, tip):
        c = slicer.qMRMLNodeComboBox()
        c.nodeTypes = types; c.selectNodeUponCreation = True; c.addEnabled = False; c.removeEnabled = False
        c.noneEnabled = True; c.showHidden = False; c.showChildNodeTypes = False
        c.setMRMLScene(slicer.mrmlScene); c.toolTip = tip
        return c

    def onStateChanged(self, _):
        assess = self.stateCombo.currentData == "assess"
        for w in (self.sizeInSitu, self.obsDepth, self.obsSd, self.retarget):
            w.enabled = assess
        for w in (self.sizes, self.depths):
            w.enabled = not assess

    def onPlacePoints(self):
        node = self.pointsSelector.currentNode() or self.logic.newPointList()
        self.pointsSelector.setCurrentNode(node)
        self.logic.startPlacing(node, multi=True)

    def onPlaceMS(self):
        node = self.msSelector.currentNode() or slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "Membranous septum")
        self.msSelector.setCurrentNode(node)
        self.logic.startPlacing(node, multi=False)

    def _floats(self, text):
        return tuple(float(t) for t in text.replace(";", ",").split(",") if t.strip())

    def onDecide(self):
        try:
            with slicer.util.tryWithErrorDisplay("Decision failed.", waitCursor=True):
                params = dict(state=self.stateCombo.currentData, sizes=tuple(int(v) for v in self._floats(self.sizes.text)),
                              depths=self._floats(self.depths.text), size_in_situ=self.sizeInSitu.value,
                              observed_depth=self.obsDepth.value, observed_sd=self.obsSd.value,
                              retarget=self._floats(self.retarget.text), eta=self.eta.value, ms_sd=self.msSd.value)
                out = self.logic.run(self.ctSelector.currentNode(), self.segSelector.currentNode(),
                                     self.pointsSelector.currentNode(), self.msSelector.currentNode(), **params)
                self._markdown = out["markdown"]
                self.report.setMarkdown(self._markdown)
        except Exception:
            pass

    def onSave(self):
        if not self._markdown:
            return
        path = qt.QFileDialog.getSaveFileName(None, "Save report", "tavr_decide_report.md", "Markdown (*.md)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self._markdown)


# -----------------------------------------------------------------------------------------
class TAVRDecideLogic(ScriptedLoadableModuleLogic):
    """Converts Slicer nodes to the twin's objects and runs the decision."""

    # ---- nodes -> twin --------------------------------------------------------------------
    @staticmethod
    def volume_from_node(node):
        from tavr_decide.geometry import Volume
        if node is None:
            raise ValueError("select a CT volume")
        if node.GetParentTransformNode() is not None:
            raise ValueError("the CT is under a transform: harden it first (Data module, right click, Harden transform)")
        m = vtk.vtkMatrix4x4(); node.GetIJKToRASMatrix(m)
        data = np.transpose(slicer.util.arrayFromVolume(node), (2, 1, 0)).astype(np.float32)   # (k,j,i) -> (i,j,k)
        return Volume(data, slicer.util.arrayFromVTKMatrix(m))

    @staticmethod
    def lumen_from_segmentation(seg_node, ref_node):
        from tavr_decide.geometry import Volume
        if seg_node is None:
            raise ValueError("select the lumen segmentation")
        ids = vtk.vtkStringArray(); seg_node.GetSegmentation().GetSegmentIDs(ids)
        if ids.GetNumberOfValues() == 0:
            raise ValueError("the segmentation has no segments")
        union = None
        for i in range(ids.GetNumberOfValues()):
            a = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, ids.GetValue(i), ref_node)
            union = a > 0 if union is None else (union | (a > 0))
        m = vtk.vtkMatrix4x4(); ref_node.GetIJKToRASMatrix(m)
        return Volume(np.transpose(union, (2, 1, 0)).astype(np.float32), slicer.util.arrayFromVTKMatrix(m))

    @staticmethod
    def landmarks_from_markups(points, ms_line, ms_sd=1.0):
        from tavr_decide.geometry import Landmarks
        if points is None or points.GetNumberOfControlPoints() < 3:
            raise ValueError("place at least the three annulus nadirs")
        pts, labels = [], []
        for i in range(points.GetNumberOfControlPoints()):
            p = [0.0, 0.0, 0.0]; points.GetNthControlPointPositionWorld(i, p)
            pts.append(tuple(p)); labels.append(points.GetNthControlPointLabel(i).lower())
        named = {l: p for l, p in zip(labels, pts)}
        nadirs = [p for l, p in zip(labels, pts) if l.startswith(("nadir", "hinge"))]
        if len(nadirs) < 3:                                   # unlabeled: take the placement order
            nadirs = pts[:3]
        lcc = named.get("lcc", pts[3] if len(pts) > 3 else None)
        rcc = named.get("rcc", pts[4] if len(pts) > 4 else None)
        if ms_line is None or ms_line.GetNumberOfControlPoints() < 2:
            raise ValueError("measure the membranous septum with a line (required: it drives the conduction holon)")
        return Landmarks(nadirs=nadirs, lcc_ostium=lcc, rcc_ostium=rcc,
                         ms_length_mm=float(ms_line.GetLineLengthWorld()), ms_length_sd_mm=float(ms_sd))

    # ---- placement helpers ---------------------------------------------------------------
    @staticmethod
    def newPointList():
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "TAVR landmarks")
        node.SetControlPointLabelFormat("%N-%d")
        observer = {}

        def relabel(caller, event):
            for i in range(caller.GetNumberOfControlPoints()):
                if i < len(LANDMARK_LABELS) and not caller.GetNthControlPointLabel(i).startswith(tuple(LANDMARK_LABELS)):
                    caller.SetNthControlPointLabel(i, LANDMARK_LABELS[i])
        observer["tag"] = node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, relabel)
        return node

    @staticmethod
    def startPlacing(node, multi):
        sel = slicer.app.applicationLogic().GetSelectionNode()
        sel.SetReferenceActivePlaceNodeClassName(node.GetClassName())
        sel.SetActivePlaceNodeID(node.GetID())
        slicer.app.applicationLogic().GetInteractionNode().SetPlaceModePersistence(1 if multi else 0)
        slicer.app.applicationLogic().GetInteractionNode().SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.Place)

    # ---- the decision --------------------------------------------------------------------
    def run(self, ct_node, seg_node, points_node, ms_node, state="position", sizes=(26, 29), depths=(3.0, 5.0),
            size_in_situ=26, observed_depth=4.0, observed_sd=0.8, retarget=(3.0, 4.0, 5.0), eta=0.95, ms_sd=1.0,
            show_plane=True, n=4000):
        from tavr_decide import Uncertain, Utility, evolut_like_grammar, holarchic_select, margin_certificate, render_markdown
        from tavr_decide.cli import default_holons
        from tavr_decide.geometry import build_anatomy, fit_plane
        ct = self.volume_from_node(ct_node)
        lumen = self.lumen_from_segmentation(seg_node, ct_node)
        lm = self.landmarks_from_markups(points_node, ms_node, ms_sd)
        anat, prov = build_anatomy(ct, lumen, lm, label=ct_node.GetName())
        if state == "assess":
            anat = replace(anat, observed_depth_mm=Uncertain(float(observed_depth), float(observed_sd)))
            grammar = evolut_like_grammar(size_in_situ=int(size_in_situ), retarget_depths=tuple(retarget))
        else:
            grammar = evolut_like_grammar(sizes=tuple(sizes), depths=tuple(depths))
        holons, utility = default_holons(), Utility()
        res, trace = holarchic_select(anat, grammar, state, holons, utility, eta=eta, n=n, seed=0)
        certs = [margin_certificate(anat, grammar, state, holons, utility, v, s, st, n=min(n, 2000))
                 for v, s, st in (("ms_length_mm", 4.0, 0.25), ("upper_lvot_calcium_mm3", 40.0, 2.0),
                                  ("annulus_diameter_mm", 2.0, 0.1))]
        if state == "assess":
            certs.append(margin_certificate(anat, grammar, state, holons, utility, "observed_depth_mm", 3.0, 0.1, n=min(n, 2000)))
        geo = ("## Geometry\n\n| measurement | value |\n|---|---|\n"
               f"| annulus diameter (area-derived) | {anat.annulus_diameter_mm.mean:.1f} ± {anat.annulus_diameter_mm.sd:.1f} mm |\n"
               f"| annulus perimeter | {prov['annulus']['perimeter_mm']:.1f} mm |\n"
               f"| upper-LVOT calcium | {anat.upper_lvot_calcium_mm3.mean:.0f} mm³ |\n"
               f"| membranous septum | {anat.ms_length_mm.mean:.1f} ± {anat.ms_length_mm.sd:.1f} mm |\n"
               f"| LCC / RCC height | {anat.lcc_coronary_height_mm.mean:.1f} / {anat.rcc_coronary_height_mm.mean:.1f} mm |\n\n")
        md = render_markdown(res, certificates=certs, trace=trace, title=f"TAVR Decide — {ct_node.GetName()}")
        md = md.replace("## Recommendation", geo + "## Recommendation", 1)
        if show_plane:
            plane = fit_plane(np.array(lm.nadirs), toward=np.array(prov["plane_center_ras"]) + 10 * np.array(prov["plane_normal_ras"]))
            self.showPlane(plane.center, plane.normal)
        return {"anatomy": anat, "provenance": prov, "result": res, "trace": trace, "certificates": certs, "markdown": md}

    @staticmethod
    def showPlane(center, normal, size=40.0):
        node = slicer.mrmlScene.GetFirstNodeByName("Annulus plane")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", "Annulus plane")
        node.SetCenterWorld(*[float(v) for v in center])
        node.SetNormalWorld(*[float(v) for v in normal])
        try:
            node.SetSizeMode(slicer.vtkMRMLMarkupsPlaneNode.SizeModeAbsolute)
            node.SetSize(size, size)
        except Exception:
            pass
        node.SetLocked(True)
        return node


# -----------------------------------------------------------------------------------------
class TAVRDecideTest(ScriptedLoadableModuleTest):
    """Phantom end to end: cylindrical lumen with a calcium nodule, landmarks placed in code."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_phantom_end_to_end()

    def _phantom(self):
        sp, shape, R = 0.5, (120, 120, 160), 11.0
        i, j, k = np.indices(shape)
        x, y, z = i * sp, j * sp, k * sp
        cx, cy, z_ann = shape[0] * sp / 2, shape[1] * sp / 2, 80 * sp
        r = np.hypot(x - cx, y - cy)
        lumen = (r <= R)
        ct = np.where(lumen, 300.0, 50.0).astype(np.float32)
        ang = np.arctan2(y - cy, x - cx)
        ct[(r > R) & (r <= R + 1.5) & (z >= z_ann - 2.5) & (z < z_ann - 0.5) & (np.abs(ang) < np.deg2rad(15))] = 1200.0
        return ct, lumen, (cx, cy, z_ann, R, sp)

    def test_phantom_end_to_end(self):
        ct, lumen, (cx, cy, z_ann, R, sp) = self._phantom()
        ct_node = slicer.util.addVolumeFromArray(np.transpose(ct, (2, 1, 0)).copy(), name="phantom CT")
        ct_node.SetSpacing(sp, sp, sp); ct_node.SetOrigin(0, 0, 0)
        ct_node.SetIJKToRASDirections(1, 0, 0, 0, 1, 0, 0, 0, 1)
        lab = slicer.util.addVolumeFromArray(np.transpose(lumen.astype(np.uint8), (2, 1, 0)).copy(),
                                             name="lumen", nodeClassName="vtkMRMLLabelMapVolumeNode")
        lab.SetSpacing(sp, sp, sp); lab.SetOrigin(0, 0, 0); lab.SetIJKToRASDirections(1, 0, 0, 0, 1, 0, 0, 0, 1)
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "lumen seg")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lab, seg)
        pts = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "TAVR landmarks")
        for t, name in zip((0.0, 2.094, 4.189), LANDMARK_LABELS[:3]):
            pts.AddControlPoint(cx + R * np.cos(t), cy + R * np.sin(t), z_ann, name)
        pts.AddControlPoint(cx - R, cy, z_ann + 13.0, "LCC")
        pts.AddControlPoint(cx, cy + R, z_ann + 15.5, "RCC")
        ms = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "MS")
        ms.AddControlPoint(cx, cy, z_ann - 4.2); ms.AddControlPoint(cx, cy, z_ann)
        out = TAVRDecideLogic().run(ct_node, seg, pts, ms, state="assess", size_in_situ=26, observed_depth=4.6, n=1500)
        a = out["anatomy"]
        assert abs(a.annulus_diameter_mm.mean - 22.0) < 0.6, a.annulus_diameter_mm
        assert abs(a.ms_length_mm.mean - 4.2) < 1e-3
        assert abs(a.lcc_coronary_height_mm.mean - 13.0) < 0.3
        assert "Recommended action" in out["markdown"] and "## Geometry" in out["markdown"]
        assert slicer.mrmlScene.GetFirstNodeByName("Annulus plane") is not None
        self.delayDisplay("TAVR Decide phantom test passed")
        return out
