"""Run inside 3D Slicer (headless):
    Slicer --no-splash --no-main-window --additional-module-paths <repo>/slicer/TAVRDecide
           --python-script run_slicer_test.py [--physics] [--segment=<ct.nii.gz>] <out.txt>
Always: builds the panel, runs the phantom case through the logic and through the Decide
button, saves a screenshot. --physics: launches the physics job from the panel (finite
elements, no CFD, review mode, one recapture depth) and waits for its report.
--segment: loads a CT, launches the segmentation job from the panel and checks that a
non-empty lumen segmentation comes back. Writes PASS/FAIL lines to <out.txt>, then exits."""
import sys
import time
import traceback

import numpy as np
import slicer

out_path = sys.argv[-1]
want_physics = "--physics" in sys.argv
seg_ct = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--segment=")), None)
lines = []
t0 = time.time()


def wait_for(timer_owner, attr, timeout_s):
    """Spin the Qt event loop until the panel's job timer stops (the job finished)."""
    t = time.time()
    timer = getattr(timer_owner, attr)
    while timer.isActive() and time.time() - t < timeout_s:
        slicer.app.processEvents()
        time.sleep(0.5)
    return not timer.isActive()


try:
    import TAVRDecide
    widget = slicer.modules.tavrdecide.widgetRepresentation()
    lines.append(f"widget built: {type(widget).__name__}")
    test = TAVRDecide.TAVRDecideTest()
    test.setUp()
    out = test.test_phantom_end_to_end()
    lines.append("PASS phantom end-to-end (logic)")
    lines.append(f"recommended: {out['result'].best.key}  stability {out['result'].pea:.3f}")
    w = widget.self()
    w.ctSelector.setCurrentNode(slicer.util.getNode("phantom CT"))
    w.segSelector.setCurrentNode(slicer.util.getNode("lumen seg"))
    w.pointsSelector.setCurrentNode(slicer.util.getNode("TAVR landmarks"))
    w.msSelector.setCurrentNode(slicer.util.getNode("MS"))
    w.stateCombo.setCurrentIndex(1)
    w.obsDepth.value = 4.6
    w.onDecide()
    shown = w.report.toPlainText()
    ok = "Recommended action" in shown and "Resumen clínico" in shown
    lines.append("PASS panel Decide button" if ok else "FAIL panel Decide button")
    png = out_path.replace(".txt", "_panel.png")
    widget.resize(460, 1500)
    widget.grab().save(png)
    lines.append(f"panel screenshot: {png}")

    if want_physics:
        w.retarget.text = "3"
        w.withCfd.checked = False
        w.reviewMode.checked = True
        w.onRunPhysics()
        lines.append(f"physics job started: {w._job['workdir'] if w._job else None}")
        finished = wait_for(w, "physicsTimer", 3600)
        status = w.physicsStatus.text
        ok = finished and status.startswith("Done") and "Physics per action" in w.report.toPlainText()
        lines.append(("PASS" if ok else "FAIL") + f" panel physics job: {status}")

    if seg_ct:
        node = slicer.util.loadVolume(seg_ct)
        w.ctSelector.setCurrentNode(node)
        w.licence.text = ""
        w.onSegment()
        finished = wait_for(w, "segTimer", 3600)
        seg = w.segSelector.currentNode()
        n_seg = seg.GetSegmentation().GetNumberOfSegments() if seg else 0
        vox = 0
        if seg and n_seg:
            sid = seg.GetSegmentation().GetNthSegmentID(0)
            vox = int((slicer.util.arrayFromSegmentBinaryLabelmap(seg, sid, node) > 0).sum())
        ok = finished and n_seg > 0 and vox > 1000
        lines.append(("PASS" if ok else "FAIL") + f" panel segmentation: {w.segStatus.text} | segments {n_seg}, voxels {vox}")
except Exception:
    lines.append("FAIL")
    lines.append(traceback.format_exc())
lines.append(f"elapsed {time.time() - t0:.1f} s")
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
slicer.util.exit(0)
