"""Focused verification for the TREC-Route browser progress model."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from pipeline_ui import PAGE, Runner, STEPS


passed = failed = 0


def check(condition: bool, label: str) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}")


def wait_for_runner(runner: Runner, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while runner.is_running() and time.monotonic() < deadline:
        time.sleep(0.02)
    if runner.is_running():
        raise TimeoutError("synthetic UI runner did not finish")


print("T1: page layout")
check("TREC-Route Pipeline" in PAGE, "UI uses the current TREC-Route name")
check("progress-wrap" in PAGE and "progress-${n}" in PAGE,
      "each generated step row has a right-side progress bar")
check("role=\"progressbar\"" in PAGE and "aria-valuenow" in PAGE,
      "progress bars expose accessible state")
check(len(STEPS) == 5, "all five top-level pipeline steps represented")
check(STEPS[2][2] == ["05", "05A", "05B", "05FACET"],
      "merged step retains all four independently skippable substages")


print("\nT2: real counter parsing")
runner = Runner(".")
runner.planned_steps = {2, 3, 4, 5}
runner.active_step = 3
for step in runner.planned_steps:
    runner.progress[step] = {"percent": 0.0, "state": "queued", "message": "Waiting"}
runner.progress[3]["state"] = "running"
runner.active_step = 2
runner.progress[2]["state"] = "running"
hidden = runner._parse_progress(
    "[osm_material_progress] percent=65 message=Terrain IDs assigned")
check(hidden and 80.0 < runner.progress[2]["percent"] < 82.0,
      "OSM terrain progress uses internal partition milestones")
runner.active_step = 3
runner._parse_progress("  SVF batch 6/12 (12000/23565 points)")
svf_percent = runner.progress[3]["percent"]
runner._parse_progress(
    "[trec_progress] step=3 percent=68 state=running message=Loading final MRT")
runner._parse_progress("  step 72/144 (11:50) -- 2s elapsed")
check(9.0 < svf_percent < 11.0, "prep SVF progress uses batch fraction")
check(82.0 < runner.progress[3]["percent"] < 83.0,
      "final MRT progress uses timestep fraction")
runner._parse_progress(
    "[trec_progress] step=3 percent=20 state=running message=Facet selection")
runner._parse_progress("  batch 10/20 -- 9s elapsed")
check(runner.progress[3]["percent"] > 82.0,
      "late facet counters cannot move merged progress backward")
runner = Runner(".")
runner.planned_steps = {3}
runner.active_step = 3
runner.progress[3] = {"percent": 45.0, "state": "running", "message": "Energy"}
runner._parse_progress("  cycle 2/3: max |dT| vs previous cycle end = 0.5 K")
check(59.0 < runner.progress[3]["percent"] < 61.0,
      "surface-energy progress uses completed cycle fraction")


print("\nT3: explicit workflow markers and monotonicity")
runner._parse_progress(
    "[trec_progress] step=3 percent=0 state=running message=Loading MRT")
runner._parse_progress("  SVF batch 12/12 (23565/23565 points)")
before = runner.progress[3]["percent"]
runner._parse_progress("  SVF batch 1/12 (2000/23565 points)")
check(runner.active_step == 3 and runner.progress[3]["state"] == "running",
      "explicit marker activates the correct row")
check(runner.progress[3]["percent"] == before,
      "late or repeated log lines cannot move a bar backward")
runner._parse_progress(
    "[trec_progress] step=3 percent=100 state=done message=Complete")
check(runner.progress[3]["percent"] == 100.0
      and runner.progress[3]["state"] == "done",
      "completion marker reaches exactly 100 percent")


print("\nT4: real subprocess lifecycle")
with tempfile.TemporaryDirectory() as tmp:
    script = Path(tmp) / "start.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "echo '[trec_progress] step=4 percent=0 state=running message=Starting plots'\n"
        "echo 'Building static key-times overview...'\n"
        "echo '[trec_progress] step=4 percent=100 state=done message=Plots complete'\n"
        "echo '[trec_progress] step=5 percent=0 state=running message=Starting routes'\n"
        "echo '  Loaded 3 routes'\n"
        "echo 'Computing UTCI along each route'\n"
        "echo '  Route 1: complete'\n"
        "echo '  Route 2: complete'\n"
        "echo '  Route 3: complete'\n"
        "echo '[trec_progress] step=5 percent=100 state=done message=Routes complete'\n",
        encoding="utf-8")
    runner = Runner(tmp)
    ok, _ = runner.start(4, False, "")
    wait_for_runner(runner)
    _lines, _offset, status, _current, progress = runner.snapshot(0)
    check(ok and status == "done", "successful child process updates overall status")
    check(progress["4"]["state"] == "done" and progress["4"]["percent"] == 100.0,
          "first planned step completes")
    check(progress["5"]["state"] == "done" and progress["5"]["percent"] == 100.0,
          "subsequent planned step completes independently")
    check(progress["1"]["state"] == "idle", "steps outside the run remain untouched")


print("\nT5: only-this fencing for every top-level step")
with tempfile.TemporaryDirectory() as tmp:
    script = Path(tmp) / "start.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    for selected, _label, _flags in STEPS:
        runner = Runner(tmp)
        ok, _ = runner.start(selected, True, "")
        wait_for_runner(runner)
        lines, _offset, status, _current, _progress = runner.snapshot(0)
        fence_line = next((line for line in lines if "fencing to one step" in line), "")
        expected = {
            f"SKIP_{flag}=1"
            for number, _later_label, flags in STEPS if number > selected
            for flag in flags
        }
        check(ok and status == "done" and all(item in fence_line for item in expected),
              f"step {selected} fences every later stage ({len(expected)} flags)")

print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
