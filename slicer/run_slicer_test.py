"""Run inside 3D Slicer (headless):
    Slicer --no-splash --no-main-window --additional-module-paths <repo>/slicer/TAVRDecide --python-script run_slicer_test.py <out.txt>
Writes PASS/FAIL, the module's report and timing to <out.txt>, then exits Slicer."""
import sys
import time
import traceback

import slicer

out_path = sys.argv[-1]
lines = []
t0 = time.time()
try:
    import TAVRDecide
    widget = slicer.modules.tavrdecide.widgetRepresentation()          # builds the panel (UI smoke test)
    lines.append(f"widget built: {type(widget).__name__}")
    test = TAVRDecide.TAVRDecideTest()
    test.setUp()
    out = test.test_phantom_end_to_end()
    lines.append("PASS phantom end-to-end")
    lines.append(f"recommended: {out['result'].best.key}  stability {out['result'].pea:.3f}")
    # drive the panel the way a user would: pick nodes, set the state, press Decide
    w = widget.self()
    w.ctSelector.setCurrentNode(slicer.util.getNode("phantom CT"))
    w.segSelector.setCurrentNode(slicer.util.getNode("lumen seg"))
    w.pointsSelector.setCurrentNode(slicer.util.getNode("TAVR landmarks"))
    w.msSelector.setCurrentNode(slicer.util.getNode("MS"))
    w.stateCombo.setCurrentIndex(1)
    w.obsDepth.value = 4.6
    w.onDecide()
    shown = w.report.toPlainText()
    lines.append("PASS panel Decide button" if "Recommended action" in shown else "FAIL panel Decide button: report empty")
    png = out_path.replace(".txt", "_panel.png")
    widget.resize(460, 1250)
    widget.grab().save(png)
    lines.append(f"panel screenshot: {png}")
    lines.append("--- report ---")
    lines.append(out["markdown"])
except Exception:
    lines.append("FAIL")
    lines.append(traceback.format_exc())
lines.append(f"elapsed {time.time() - t0:.1f} s")
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
slicer.util.exit(0)
