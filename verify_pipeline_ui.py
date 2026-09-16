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
check("Problem selection" in PAGE and "inputcase" in PAGE and "outputcase" in PAGE,
      "Step 1 selects input and output case folders")
check("Generate routes" not in PAGE and "Geometry build" not in PAGE,
      "case-specific preparation actions are absent from the UI")
check(len(STEPS) == 5 and [step[0] for step in STEPS] == [2, 3, 4, 5, 6],
      "only executable steps 2-6 have run controls")
check(STEPS[1][2] == ["PEDESTRIAN_WIND"],
      "step 3 owns the pedestrian potential-flow wind stage")
check(STEPS[2][2] == ["05", "05A", "05B", "05FACET"],
      "merged MRT step (now 4) retains all four independently skippable substages")


print("\nT2: real counter parsing")
runner = Runner(".")
runner.planned_steps = {2, 3, 4, 5, 6}
for step in runner.planned_steps:
    runner.progress[step] = {"percent": 0.0, "state": "queued", "message": "Waiting"}
runner.active_step = 2
runner.progress[2]["state"] = "running"
hidden = runner._parse_progress(
    "[osm_material_progress] percent=65 message=Terrain IDs assigned")
check(hidden and 80.0 < runner.progress[2]["percent"] < 82.0,
      "OSM terrain progress uses internal partition milestones")

runner.active_step = 3
runner.progress[3]["state"] = "running"
runner._parse_progress("  cut-cell occupancy batch 5/10")
check(32.0 < runner.progress[3]["percent"] < 33.0,
      "potential-flow geometry sampling uses batch fraction")
runner._parse_progress("Potential-flow solve complete: relative residual 1e-12")
check(84.0 < runner.progress[3]["percent"] < 86.0,
      "potential-flow solve completion marker advances step 3")

runner.active_step = 4
runner.progress[4]["state"] = "running"
runner._parse_progress("  SVF batch 6/12 (12000/23565 points)")
svf_percent = runner.progress[4]["percent"]
runner._parse_progress(
    "[trec_progress] step=4 percent=68 state=running message=Loading final MRT")
runner._parse_progress("  step 72/144 (11:50) -- 2s elapsed")
check(9.0 < svf_percent < 11.0, "prep SVF progress uses batch fraction")
check(82.0 < runner.progress[4]["percent"] < 83.0,
      "final MRT progress uses timestep fraction")
runner._parse_progress(
    "[trec_progress] step=4 percent=20 state=running message=Facet selection")
runner._parse_progress("  batch 10/20 -- 9s elapsed")
check(runner.progress[4]["percent"] > 82.0,
      "late facet counters cannot move merged progress backward")
runner = Runner(".")
runner.planned_steps = {4}
runner.active_step = 4
runner.progress[4] = {"percent": 45.0, "state": "running", "message": "Energy"}
runner._parse_progress("  cycle 2/3: max |dT| vs previous cycle end = 0.5 K")
check(59.0 < runner.progress[4]["percent"] < 61.0,
      "surface-energy progress uses completed cycle fraction")


print("\nT3: explicit workflow markers and monotonicity")
runner._parse_progress(
    "[trec_progress] step=4 percent=0 state=running message=Loading MRT")
runner._parse_progress("  SVF batch 12/12 (23565/23565 points)")
before = runner.progress[4]["percent"]
runner._parse_progress("  SVF batch 1/12 (2000/23565 points)")
check(runner.active_step == 4 and runner.progress[4]["state"] == "running",
      "explicit marker activates the correct row")
check(runner.progress[4]["percent"] == before,
      "late or repeated log lines cannot move a bar backward")
runner._parse_progress(
    "[trec_progress] step=4 percent=100 state=done message=Complete")
check(runner.progress[4]["percent"] == 100.0
      and runner.progress[4]["state"] == "done",
      "completion marker reaches exactly 100 percent")


print("\nT4: real subprocess lifecycle")
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "input" / "MMC").mkdir(parents=True)
    (Path(tmp) / "input" / "MMC" / "case.json").write_text("{}")
    script = Path(tmp) / "start.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "echo '[trec_progress] step=5 percent=0 state=running message=Starting plots'\n"
        "echo 'Building static key-times overview...'\n"
        "echo '[trec_progress] step=5 percent=100 state=done message=Plots complete'\n"
        "echo '[trec_progress] step=6 percent=0 state=running message=Starting routes'\n"
        "echo '  Loaded 3 routes'\n"
        "echo 'Computing UTCI along each route'\n"
        "echo '  Route 1: complete'\n"
        "echo '  Route 2: complete'\n"
        "echo '  Route 3: complete'\n"
        "echo '[trec_progress] step=6 percent=100 state=done message=Routes complete'\n",
        encoding="utf-8")
    runner = Runner(tmp)
    ok, _ = runner.start(5, False, "")
    wait_for_runner(runner)
    _lines, _offset, status, _current, progress = runner.snapshot(0)
    check(ok and status == "done", "successful child process updates overall status")
    check(progress["5"]["state"] == "done" and progress["5"]["percent"] == 100.0,
          "first planned step completes")
    check(progress["6"]["state"] == "done" and progress["6"]["percent"] == 100.0,
          "subsequent planned step completes independently")
    check(progress["2"]["state"] == "idle", "steps outside the run remain untouched")


print("\nT5: only-this fencing for every top-level step")
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "input" / "MMC").mkdir(parents=True)
    (Path(tmp) / "input" / "MMC" / "case.json").write_text("{}")
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

print("\nT6: no retired optional-solver activation")
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "input" / "MMC").mkdir(parents=True)
    (Path(tmp) / "input" / "MMC" / "case.json").write_text("{}")
    script = Path(tmp) / "start.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "echo WITH_MICROCLIMATE=${WITH_MICROCLIMATE:-unset}\n"
        "exit 0\n", encoding="utf-8")
    runner = Runner(tmp)
    ok, _ = runner.start(4, True, "")
    wait_for_runner(runner)
    lines, _offset, status, _current, _progress = runner.snapshot(0)
    check(ok and status == "done" and "WITH_MICROCLIMATE=unset" in lines,
          "the UI no longer force-enables the retired 3-D microclimate solver")

print("\nT7: run-all-cases batch mode")
check("Run all cases" in PAGE and "allstep" in PAGE and "runAll()" in PAGE,
      "page offers the run-all control with a from-step dropdown")
with tempfile.TemporaryDirectory() as tmp:
    for name in ("alpha", "beta", "gamma"):
        (Path(tmp) / "input" / name).mkdir(parents=True)
        (Path(tmp) / "input" / name / "case.json").write_text("{}")
    script = Path(tmp) / "start.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"CASE=$CASE_NAME STEP=$1 OUT=$OUTPUT_CASE_DIR\"\n"
        "[ \"$CASE_NAME\" = beta ] && exit 3\n"
        "exit 0\n", encoding="utf-8")
    runner = Runner(tmp)
    ok, msg = runner.start_all(4, "")
    wait_for_runner(runner, timeout=10.0)
    lines, _offset, status, current, _progress = runner.snapshot(0)
    text = "\n".join(lines)
    check(ok and "3 case(s)" in msg, "batch accepts every discovered case")
    check(all(f"CASE={name} STEP=4" in text
              for name in ("alpha", "beta", "gamma")),
          "every case ran sequentially from the chosen step")
    check(f"OUT={tmp}/run_output/alpha" in text,
          "each case writes into its own run_output folder")
    check(status == "failed" and "beta (exit code 3)" in text,
          "a failing case is reported by name")
    check("CASE=gamma" in text.split("exited with code 3")[-1],
          "remaining cases continue after a failure")
    check("2/3 succeeded" in current,
          "final status summarises successes and failures")

    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    runner = Runner(tmp)
    ok, _ = runner.start_all(5, "")
    wait_for_runner(runner, timeout=10.0)
    _lines, _offset, status, current, _progress = runner.snapshot(0)
    check(ok and status == "done" and "All 3 cases completed" in current,
          "an all-successful batch finishes as done")
    ok, msg = Runner(tmp).start_all(9, "")
    check(not ok and "Invalid" in msg, "batch rejects an unknown step")

print("\nT8: developer / post-processing section")
from pipeline_ui import DEV_TOOLS
check("__TOOLS__" in PAGE and "runTool(" in PAGE and "devpanel" in PAGE,
      "page renders a developer tool panel")
check("temporary" in PAGE.lower() and "not part of the numbered pipeline" in PAGE,
      "the panel is labelled as temporary and outside the pipeline")
check({"validation_radiometer", "compare_solweig", "validate_routes"}
      <= set(DEV_TOOLS),
      "registry covers validation, cross-model and preprocessing tools")
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "input" / "MMC").mkdir(parents=True)
    (Path(tmp) / "input" / "MMC" / "case.json").write_text("{}")
    runner = Runner(tmp)
    ok, msg = runner.start_tool("rm -rf /", "")
    check(not ok and "Unknown developer tool" in msg,
          "an arbitrary command string is refused (registry keys only)")
    ok, msg = runner.start_tool("not_a_tool", "")
    check(not ok and "Unknown developer tool" in msg,
          "an unknown registry key is refused")
    for key, (label, description, builder) in DEV_TOOLS.items():
        argv = builder({"input_dir": "/case/in", "output_dir": "/case/out",
                        "input_case": "MMC", "output_case": "MMC"})
        check(isinstance(argv, list) and argv[0] == "python3"
              and Path(argv[1]).suffix == ".py"
              and (Path(tmp).parent / argv[1]).name == argv[1],
              f"tool '{key}' builds a python argv list ({argv[1]})")
    # A tool that exits nonzero must surface as failed, not as a pipeline step.
    script = Path(tmp) / "compare_mrt_lisbon_data.py"
    script.write_text("import sys\nprint('tool ran')\nsys.exit(3)\n",
                      encoding="utf-8")
    runner = Runner(tmp)
    ok, _ = runner.start_tool("validation_radiometer", "")
    wait_for_runner(runner, timeout=10.0)
    lines, _offset, status, current, progress = runner.snapshot(0)
    check(ok and status == "failed" and "tool ran" in lines,
          "a developer tool's output is captured and its failure reported")
    check(all(item["state"] == "idle" for item in progress.values()),
          "developer tools never touch the pipeline step progress bars")

print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
