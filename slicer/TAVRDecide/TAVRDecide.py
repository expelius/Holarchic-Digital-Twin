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
        self.licence = qt.QLineEdit(qt.QSettings().value("TAVRDecide/TotalSegmentatorLicence", ""))
        self.licence.placeholderText = "optional: TotalSegmentator academic licence (free) -> LVOT, cusps, automatic nadirs"
        form.addRow("Licence:", self.licence)
        self.segButton = qt.QPushButton("Segment automatically (background)")
        self.segButton.toolTip = ("Runs TotalSegmentator on the CT. Without licence: aorta only. With the free academic "
                                  "licence: aorta, LV, LVOT, cusps, annulus, and the three nadirs placed automatically.")
        self.segButton.connect("clicked()", self.onSegment); form.addRow(self.segButton)
        self.segStatus = qt.QLabel(""); self.segStatus.wordWrap = True; form.addRow(self.segStatus)
        self.segTimer = qt.QTimer(); self.segTimer.setInterval(4000); self.segTimer.connect("timeout()", self.onPollSegment)
        self._segJob = None

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
        self.langCombo = qt.QComboBox(); self.langCombo.addItem("Español", "es"); self.langCombo.addItem("English", "en")
        form.addRow("Report language:", self.langCombo)
        self.onStateChanged(0)

        # 4. decide
        self.decideButton = qt.QPushButton("Decide"); self.decideButton.setStyleSheet("font-weight: bold; padding: 6px")
        self.decideButton.connect("clicked()", self.onDecide); self.layout.addWidget(self.decideButton)
        # 5. physics in the background
        box = ctk.ctkCollapsibleButton(); box.text = "5. Physics: finite elements + CFD (background)"; box.collapsed = True
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.withCfd = qt.QCheckBox("Include paravalvular CFD (about 20 min per action)"); self.withCfd.checked = True
        form.addRow(self.withCfd)
        self.reviewMode = qt.QCheckBox("Review mode: run the physics for every action, not only when the decision needs it")
        form.addRow(self.reviewMode)
        self.physicsButton = qt.QPushButton("Run physics in background"); self.physicsButton.connect("clicked()", self.onRunPhysics)
        form.addRow(self.physicsButton)
        self.physicsStatus = qt.QLabel("Not started."); self.physicsStatus.wordWrap = True
        form.addRow("Status:", self.physicsStatus)
        self.physicsTimer = qt.QTimer(); self.physicsTimer.setInterval(5000); self.physicsTimer.connect("timeout()", self.onPollPhysics)
        self._job = None

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
                              retarget=self._floats(self.retarget.text), eta=self.eta.value, ms_sd=self.msSd.value,
                              lang=self.langCombo.currentData)
                out = self.logic.run(self.ctSelector.currentNode(), self.segSelector.currentNode(),
                                     self.pointsSelector.currentNode(), self.msSelector.currentNode(), **params)
                self._markdown = out["markdown"]
                self.report.setMarkdown(self._markdown)
        except Exception:
            pass

    def onSegment(self):
        try:
            with slicer.util.tryWithErrorDisplay("Could not start the segmentation.", waitCursor=True):
                lic = self.licence.text.strip()
                if lic:
                    qt.QSettings().setValue("TAVRDecide/TotalSegmentatorLicence", lic)
                self._segJob = self.logic.startSegment(self.ctSelector.currentNode(), licence=lic or None)
                self.segStatus.text = "Segmenting in the background (several minutes on CPU)..."
                self.segButton.enabled = False
                self.segTimer.start()
        except Exception:
            pass

    def onPollSegment(self):
        if not self._segJob:
            return
        st = self.logic.jobStatus(self._segJob)
        state = st.get("state", "starting")
        if state == "running":
            self.segStatus.text = f"Running since {self._segJob['started']}: {st.get('step', '')}"
            return
        self.segTimer.stop(); self.segButton.enabled = True
        if state == "done":
            seg, pts = self.logic.loadSegmentResult(self._segJob, self.ctSelector.currentNode())
            self.segSelector.setCurrentNode(seg)
            if pts is not None:
                self.pointsSelector.setCurrentNode(pts)
            msg = f"Done in {st.get('elapsed_s', '?')} s."
            if st.get("note"):
                msg += " " + st["note"]
            if pts is not None:
                msg += " Automatic nadirs placed: review them and add the LCC and RCC ostia."
            self.segStatus.text = msg
        else:
            self.segStatus.text = f"Segmentation failed: {st.get('message', '')} (log: {self._segJob['log']})"

    def _params(self):
        return dict(state=self.stateCombo.currentData, sizes=tuple(int(v) for v in self._floats(self.sizes.text)),
                    depths=self._floats(self.depths.text), size_in_situ=self.sizeInSitu.value,
                    observed_depth=self.obsDepth.value, observed_sd=self.obsSd.value,
                    retarget=self._floats(self.retarget.text), eta=self.eta.value, ms_sd=self.msSd.value,
                    lang=self.langCombo.currentData)

    def onRunPhysics(self):
        try:
            with slicer.util.tryWithErrorDisplay("Could not start the physics job.", waitCursor=True):
                self._job = self.logic.startPhysics(self.ctSelector.currentNode(), self.segSelector.currentNode(),
                                                    self.pointsSelector.currentNode(), self.msSelector.currentNode(),
                                                    cfd=self.withCfd.checked, review=self.reviewMode.checked,
                                                    **self._params())
                self.physicsStatus.text = f"Started. Working folder: {self._job['workdir']}"
                self.physicsButton.enabled = False
                self.physicsTimer.start()
        except Exception:
            pass

    def onPollPhysics(self):
        if not self._job:
            return
        st = self.logic.physicsStatus(self._job)
        state = st.get("state", "starting")
        if state == "running":
            self.physicsStatus.text = f"Running since {self._job['started']}: {st.get('step', '')}"
        elif state == "done":
            self.physicsTimer.stop(); self.physicsButton.enabled = True
            self.physicsStatus.text = f"Done. Recommended: {st.get('recommended')} (stability {st.get('stability', 0):.2f})."
            self._markdown = self.logic.physicsReport(self._job)
            self.report.setMarkdown(self._markdown)
        elif state == "error":
            self.physicsTimer.stop(); self.physicsButton.enabled = True
            self.physicsStatus.text = f"Error: {st.get('message', '')} (log: {self._job['log']})"
        elif state == "exited":
            self.physicsTimer.stop(); self.physicsButton.enabled = True
            self.physicsStatus.text = f"The job stopped without a result (log: {self._job['log']})"

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
            show_plane=True, n=4000, lang="es"):
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
        from tavr_decide.clinical import clinical_summary
        md = render_markdown(res, certificates=certs, trace=trace, title=f"TAVR Decide — {ct_node.GetName()}")
        md = md.replace("## Recommendation", clinical_summary(res, anat, certs, trace, lang) + "\n" + geo
                        + "## Technical detail\n\n## Recommendation", 1)
        if show_plane:
            plane = fit_plane(np.array(lm.nadirs), toward=np.array(prov["plane_center_ras"]) + 10 * np.array(prov["plane_normal_ras"]))
            self.showPlane(plane.center, plane.normal)
        return {"anatomy": anat, "provenance": prov, "result": res, "trace": trace, "certificates": certs, "markdown": md}

    # ---- physics as a background job ----------------------------------------------------
    @staticmethod
    def pythonSlicer():
        exe = "PythonSlicer.exe" if os.name == "nt" else "PythonSlicer"
        return os.path.join(slicer.app.slicerHome, "bin", exe)

    def startPhysics(self, ct_node, seg_node, points_node, ms_node, cfd=True, review=False, workdir=None,
                     state="position", sizes=(26, 29), depths=(3.0, 5.0), size_in_situ=26, observed_depth=4.0,
                     observed_sd=0.8, retarget=(3.0, 4.0, 5.0), eta=0.95, ms_sd=1.0, lang="es", **_):
        """Export the case and launch ``tavr_decide.cli physics`` as a separate process."""
        import json
        import subprocess
        import time
        import nibabel as nib
        ct = self.volume_from_node(ct_node)
        lumen = self.lumen_from_segmentation(seg_node, ct_node)
        lm = self.landmarks_from_markups(points_node, ms_node, ms_sd)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        wd = workdir or os.path.join(slicer.app.temporaryPath, "TAVRDecide", f"physics-{stamp}")
        os.makedirs(wd, exist_ok=True)
        nib.save(nib.Nifti1Image(ct.data, ct.affine), os.path.join(wd, "ct.nii.gz"))
        nib.save(nib.Nifti1Image(lumen.data.astype(np.uint8), lumen.affine), os.path.join(wd, "lumen.nii.gz"))
        with open(os.path.join(wd, "landmarks.json"), "w", encoding="utf-8") as fh:
            json.dump({"nadirs": [list(p) for p in lm.nadirs],
                       "lcc_ostium": list(lm.lcc_ostium) if lm.lcc_ostium else None,
                       "rcc_ostium": list(lm.rcc_ostium) if lm.rcc_ostium else None,
                       "ms_length_mm": lm.ms_length_mm, "ms_length_sd_mm": lm.ms_length_sd_mm}, fh)
        cmd = [self.pythonSlicer(), "-m", "tavr_decide.cli", "physics", os.path.join(wd, "ct.nii.gz"),
               "--mask", os.path.join(wd, "lumen.nii.gz"), "--landmarks", os.path.join(wd, "landmarks.json"),
               "--workdir", os.path.join(wd, "job"), "--label", ct_node.GetName(), "--state", state,
               "--eta", str(eta), "--lang", lang]
        if state == "assess":
            cmd += ["--size", str(size_in_situ), "--observed-depth", str(observed_depth), "--observed-sd", str(observed_sd),
                    "--retarget-depths", *[str(d) for d in retarget]]
        else:
            cmd += ["--sizes", *[str(s) for s in sizes], "--depths", *[str(d) for d in depths]]
        if review:
            cmd.append("--force")
        if not cfd:
            cmd.append("--no-cfd")
        log = os.path.join(wd, "physics.log")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=wd, creationflags=flags)
        return {"proc": proc, "workdir": wd, "log": log, "started": time.strftime("%H:%M:%S"), "cmd": cmd}

    def _launch(self, wd, args):
        import subprocess
        import time
        log = os.path.join(wd, "job.log")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        cmd = [self.pythonSlicer(), "-m", "tavr_decide.cli", *args]
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=wd, creationflags=flags)
        return {"proc": proc, "workdir": wd, "log": log, "started": time.strftime("%H:%M:%S"), "cmd": cmd}

    def startSegment(self, ct_node, licence=None, workdir=None, device="cpu"):
        import time
        import nibabel as nib
        ct = self.volume_from_node(ct_node)
        wd = workdir or os.path.join(slicer.app.temporaryPath, "TAVRDecide", f"segment-{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(wd, exist_ok=True)
        nib.save(nib.Nifti1Image(ct.data, ct.affine), os.path.join(wd, "ct.nii.gz"))
        args = ["segment", os.path.join(wd, "ct.nii.gz"), "--workdir", os.path.join(wd, "job"), "--device", device]
        if licence:
            args += ["--licence", licence]
        job = self._launch(wd, args)
        job["kind"] = "segment"
        return job

    @staticmethod
    def jobStatus(job):
        import json
        path = os.path.join(job["workdir"], "job", "status.json")
        st = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    st = json.load(fh)
            except (OSError, ValueError):
                st = {}
        if st.get("state") not in ("done", "error") and job["proc"].poll() is not None:
            return {"state": "error", "message": f"the job exited with code {job['proc'].returncode}"}
        return st

    def loadSegmentResult(self, job, ct_node):
        """Load the lumen as a segmentation node and, when present, the automatic nadirs."""
        import json
        st = self.jobStatus(job)
        lab = slicer.util.loadLabelVolume(st["lumen"], {"show": False})
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Lumen (TotalSegmentator)")
        seg.SetReferenceImageGeometryParameterFromVolumeNode(ct_node)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lab, seg)
        slicer.mrmlScene.RemoveNode(lab)
        pts = None
        if st.get("landmarks_auto"):
            with open(st["landmarks_auto"], encoding="utf-8") as fh:
                nad = json.load(fh)["nadirs_labelled"]
            pts = self.newPointList()
            pts.SetName("TAVR landmarks (automatic nadirs)")
            for label in ("nadir_N", "nadir_L", "nadir_R"):
                if label in nad:
                    pts.AddControlPoint(*nad[label], label)
        return seg, pts

    @staticmethod
    def physicsStatus(job):
        import json
        path = os.path.join(job["workdir"], "job", "status.json")
        st = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    st = json.load(fh)
            except (OSError, ValueError):
                st = {}
        if st.get("state") not in ("done", "error") and job["proc"].poll() is not None:
            return {"state": "exited", "returncode": job["proc"].returncode}
        return st

    @staticmethod
    def physicsReport(job):
        with open(os.path.join(job["workdir"], "job", "report.md"), encoding="utf-8") as fh:
            return fh.read()

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
        assert "Resumen clínico" in out["markdown"] and "Recapturar" in out["markdown"]
        assert slicer.mrmlScene.GetFirstNodeByName("Annulus plane") is not None
        self.delayDisplay("TAVR Decide phantom test passed")
        return out
