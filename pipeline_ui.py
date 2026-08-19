#!/usr/bin/env python3
"""
pipeline_ui.py -- local browser UI for running the TREC-Route pipeline.

WHY THIS APPROACH
-----------------
Stdlib only: http.server + a single HTML page. No pip install, no Flask,
no Streamlit, no Tkinter (which needs python3-tk and gives a worse log
view). One file, ~zero startup cost, runs in your browser.

WHAT IT DOES
------------
  * Step 1 selects an input case and its corresponding output case
  * One button per executable pipeline step (2-6) to run ONLY that step
  * A second button per step to run that step AND everything after it
  * Live logs and per-step progress measured from real workflow checkpoints
    and numerical loop counters, with a Stop button

"Run only this step" works by calling `start.sh N` with SKIP_<NAME>=1 set
for every stage after N -- start.sh itself always runs N..end, so the skips
fence it to a single step.

USAGE
-----
    python3 pipeline_ui.py                # opens http://127.0.0.1:8765
    python3 pipeline_ui.py --port 9000
    python3 pipeline_ui.py --dir "/media/harshin/data_drive/route based UTCI"

Put this file next to start.sh (or pass --dir).
"""

import argparse
import html
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# step number -> (label, [SKIP_ flags that belong to this step])
STEPS = [
    (2, "OSM data + ground materials", ["OSM", "OSM_MATERIALS"]),
    (3, "Facet-thermal MRT (prep → 05a → 05b → MRT)",
     ["05", "05A", "05B", "05FACET"]),
    (4, "Optional full-surface radiation + 3-D Ta/velocity → recoupled MRT",
     ["URBAN_RADIATION", "MICROCLIMATE", "MICROCLIMATE_05B",
      "MICROCLIMATE_05FACET"]),
    (5, "Visualizations (06, 07)", ["06", "07"]),
    (6, "Route stress (UTCI 08 + JOS-3 09 + compare 10)",
     ["08", "09", "10"]),
]
LAST_STEP = STEPS[-1][0]
PROGRESS_RE = re.compile(
    r"^\[trec_progress\]\s+step=(\d+)\s+percent=([0-9.]+)\s+"
    r"state=(\w+)\s+message=(.*)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class Runner:
    """Owns at most one child process and its captured output."""

    def __init__(self, workdir):
        self.workdir = workdir
        self.input_root = os.path.join(workdir, "input")
        self.output_root = os.path.join(workdir, "run_output")
        self.proc = None
        self.lines = []
        self.lock = threading.Lock()
        self.status = "idle"      # idle | running | done | failed | stopped
        self.current = ""
        self.active_step = None
        self.planned_steps = set()
        self.progress = self._empty_progress()
        self.route_count = 0
        self.route_phase = ""

    @staticmethod
    def _empty_progress():
        return {
            number: {"percent": 0.0, "state": "idle", "message": "Not run"}
            for number, _label, _flags in STEPS
        }

    # -- helpers ---------------------------------------------------------
    def _append(self, text):
        with self.lock:
            self.lines.append(text)

    def snapshot(self, offset):
        with self.lock:
            progress = {str(key): dict(value)
                        for key, value in self.progress.items()}
            return (self.lines[offset:], len(self.lines), self.status,
                    self.current, progress)

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    # -- run -------------------------------------------------------------
    @staticmethod
    def _case_name(name):
        name = str(name or "").strip()
        if not name or name in {".", ".."} or os.path.basename(name) != name:
            raise ValueError(f"Invalid case folder name: {name!r}")
        return name

    def available_cases(self):
        """Return selectable input cases and existing/suggested output cases."""
        input_cases = []
        if os.path.isdir(self.input_root):
            input_cases = sorted(
                entry.name for entry in os.scandir(self.input_root)
                if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "case.json")))
        output_cases = []
        if os.path.isdir(self.output_root):
            output_cases = sorted(
                entry.name for entry in os.scandir(self.output_root) if entry.is_dir())
        return input_cases, sorted(set(input_cases) | set(output_cases))

    def _resolve_cases(self, input_case, output_case):
        input_name = self._case_name(input_case)
        output_name = self._case_name(output_case)
        input_dir = os.path.abspath(os.path.join(self.input_root, input_name))
        output_dir = os.path.abspath(os.path.join(self.output_root, output_name))
        if os.path.commonpath([input_dir, os.path.abspath(self.input_root)]) != os.path.abspath(self.input_root):
            raise ValueError("Input case must be below the project input folder")
        if os.path.commonpath([output_dir, os.path.abspath(self.output_root)]) != os.path.abspath(self.output_root):
            raise ValueError("Output case must be below the project run_output folder")
        if not os.path.isfile(os.path.join(input_dir, "case.json")):
            raise ValueError(f"Input case has no case.json: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        return input_name, output_name, input_dir, output_dir

    def start(self, step, only, extra_env, input_case="MMC", output_case="MMC"):
        if self.is_running():
            return False, "A run is already in progress."
        if step not in {number for number, _label, _flags in STEPS}:
            return False, f"Invalid pipeline step: {step}"

        try:
            input_name, output_name, input_dir, output_dir = self._resolve_cases(
                input_case, output_case)
        except (OSError, ValueError) as exc:
            return False, str(exc)

        env = os.environ.copy()
        skips = []
        if only:
            # fence to a single step: skip every stage belonging to a later step
            for n, _label, flags in STEPS:
                if n > step:
                    for f in flags:
                        env[f"SKIP_{f}"] = "1"
                        skips.append(f"SKIP_{f}=1")

        # User overrides may change model settings, but case-path settings are
        # controlled exclusively by Step 1 so outputs cannot leak to another case.
        case_path_overrides = {
            "CASE_CONFIG", "ROUTES_DIR", "WEATHER_CSV", "GEOM_DIR",
            "BUILDINGS_STL", "VEGETATION_STL", "GROUND_STL", "OUT_ROOT",
            "OSM_DIR", "OSM_COMPLETE_CACHE", "OSM_GROUND_FEATURES",
            "OSM_GROUND_CONFIG", "OSM_GROUND_MATERIAL_DIR",
            "RADIANT_FLUX_CONFIG", "MRT_DIR", "THERMAL_DIR",
            "MRT_FACET_DIR", "MICROCLIMATE_DIR", "URBAN_RADIATION_DIR",
            "URBAN_RADIATION_CONFIG", "VIS_DIR", "SVF_CACHE_DIR",
            "SOLWEIG_COMPARE_OUTPUT_DIR", "GRAPHML", "POLYLINES",
            "ROUTE_POLYLINES",
        }
        applied, ignored = [], []
        for token in shlex.split(extra_env or ""):
            if "=" in token:
                k, v = token.split("=", 1)
                if k in case_path_overrides:
                    ignored.append(k)
                else:
                    env[k] = v
                    applied.append(f"{k}={v}")
        # These two settings belong to the UI transport and cannot be disabled
        # by an optional user override.
        for key in case_path_overrides:
            env.pop(key, None)
        env["PYTHONUNBUFFERED"] = "1"
        env["TREC_PROGRESS"] = "1"
        env["CASE_NAME"] = input_name
        env["INPUT_CASE_DIR"] = input_dir
        env["OUTPUT_CASE_DIR"] = output_dir
        env["OUT_ROOT"] = output_dir
        # Selecting the visibly optional step is an explicit request to run
        # it. Starting an earlier step with "This + after" leaves the
        # enhancement off unless WITH_MICROCLIMATE=1 was entered manually.
        if step == 4:
            env["WITH_MICROCLIMATE"] = "1"

        mode = "only this step" if only else "this step and everything after"
        planned = {step} if only else set(range(step, LAST_STEP + 1))
        with self.lock:
            self.lines = []
            self.status = "running"
            self.current = f"Step {step} ({mode})"
            self.active_step = step
            self.planned_steps = planned
            self.progress = self._empty_progress()
            for number in planned:
                self.progress[number] = {
                    "percent": 0.0, "state": "queued", "message": "Waiting"
                }
            self.progress[step] = {
                "percent": 0.0, "state": "running", "message": "Starting"
            }
            self.route_count = 0
            self.route_phase = ""
        self._append(f"$ cd {self.workdir}")
        self._append(f"$ input case:  {input_name} ({input_dir})")
        self._append(f"$ output case: {output_name} ({output_dir})")
        if applied:
            self._append(f"$ env {' '.join(applied)}")
        if ignored:
            self._append("$ ignored case-path override(s): " + ", ".join(ignored))
        if skips:
            self._append(f"$ (fencing to one step: {' '.join(skips)})")
        self._append(f"$ bash start.sh {step}\n")

        try:
            self.proc = subprocess.Popen(
                ["bash", "start.sh", str(step)],
                cwd=self.workdir, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
            )
        except Exception as exc:                      # e.g. start.sh missing
            with self.lock:
                self.status = "failed"
                self.progress[step]["state"] = "failed"
                self.progress[step]["message"] = "Could not start"
            self._append(f"ERROR: could not launch start.sh: {exc}")
            return False, str(exc)

        threading.Thread(target=self._pump, daemon=True).start()
        return True, "started"

    def _update_progress(self, step, percent=None, state=None, message=None,
                         allow_decrease=False):
        """Update one planned step without allowing noisy logs to move backward."""
        with self.lock:
            if step not in self.planned_steps:
                return
            item = self.progress[step]
            if percent is not None:
                bounded = max(0.0, min(100.0, float(percent)))
                item["percent"] = (bounded if allow_decrease
                                   else max(float(item["percent"]), bounded))
            if state is not None:
                item["state"] = state
            if message:
                item["message"] = message
            if state == "running":
                for prior in self.planned_steps:
                    if prior < step and self.progress[prior]["state"] == "running":
                        self.progress[prior].update(
                            percent=100.0, state="done", message="Complete")
                self.active_step = step
                self.current = f"Step {step}: {item['message']}"
            elif state in {"done", "skipped", "failed", "stopped"}:
                self.current = f"Step {step}: {item['message']}"

    def _parse_progress(self, raw_line):
        """Consume explicit shell markers and real numerical loop counters."""
        line = ANSI_RE.sub("", raw_line).strip()
        marker = PROGRESS_RE.match(line)
        if marker:
            step = int(marker.group(1))
            percent = float(marker.group(2))
            state = marker.group(3)
            message = marker.group(4).strip()
            self._update_progress(
                step, 100.0 if state in {"done", "skipped"} else percent,
                state, message, allow_decrease=(percent == 0.0))
            return True

        with self.lock:
            step = self.active_step
        if step not in self.planned_steps:
            return False

        if step == 2:
            nested = re.search(
                r"^\[osm_material_progress\]\s+percent=([0-9.]+)\s+message=(.*)$",
                line)
            if nested:
                fraction = max(0.0, min(100.0, float(nested.group(1)))) / 100.0
                self._update_progress(
                    step, 55 + 40 * fraction, "running", nested.group(2).strip())
                return True

        # The merged MRT step exposes counters from all four numerical phases.
        if step == 3:
            with self.lock:
                current_percent = float(self.progress[step]["percent"])
            match = re.search(r"SVF batch\s+(\d+)/(\d+)", line)
            if match and int(match.group(2)):
                fraction = int(match.group(1)) / int(match.group(2))
                if current_percent < 20.0:
                    self._update_progress(
                        step, 2 + 16 * fraction, "running",
                        f"Preparation: sky-view ray tracing {int(100 * fraction)}%")
                elif current_percent < 95.0:
                    self._update_progress(
                        step, 69 + 8 * fraction, "running",
                        f"Final MRT: rebuilding sky view {int(100 * fraction)}%")
            match = re.search(r"\bbatch\s+(\d+)/(\d+)", line)
            if match and int(match.group(2)) and 20.0 <= current_percent < 45.0:
                fraction = int(match.group(1)) / int(match.group(2))
                self._update_progress(
                    step, 20 + 23 * fraction, "running",
                    f"Facet visibility {int(100 * fraction)}%")
            match = re.search(r"\bcycle\s+(\d+)/(\d+)", line)
            if match and int(match.group(2)) and 45.0 <= current_percent < 68.0:
                fraction = int(match.group(1)) / int(match.group(2))
                self._update_progress(
                    step, 45 + 22 * fraction, "running",
                    f"Surface-energy cycles {int(100 * fraction)}%")
            match = re.search(r"\bstep\s+(\d+)/(\d+)", line)
            if match and int(match.group(2)):
                fraction = int(match.group(1)) / int(match.group(2))
                if 68.0 <= current_percent < 96.0:
                    self._update_progress(
                        step, 70 + 25 * fraction, "running",
                        f"Facet-thermal radiation {int(100 * fraction)}%")

        if step == 4:
            phase_markers = (
                ("Partitioning terrain triangles", 4, "Aligning triangles to material boundaries"),
                ("Precomputing full-domain facet sky-view", 8, "Full-surface sky-view ray tracing"),
                ("Precomputing ray-visible route-zone", 12, "Route-zone longwave view factors"),
                ("Reusing exact-match full-surface", 12, "Reusing full-surface radiation geometry"),
            )
            for token, percent, message in phase_markers:
                if token in line:
                    self._update_progress(step, percent, "running", message)
            match = re.search(r"full-surface radiation\s+(\d+)/(\d+)", line)
            if match and int(match.group(2)):
                fraction = int(match.group(1)) / int(match.group(2))
                self._update_progress(
                    step, 12 + 27 * fraction, "running",
                    f"Surface radiation/conduction {int(100 * fraction)}%")
            match = re.search(r"microclimate step\s+(\d+)/(\d+)", line)
            if match and int(match.group(2)):
                fraction = int(match.group(1)) / int(match.group(2))
                self._update_progress(
                    step, 40 + 31 * fraction, "running",
                    f"3-D air-field solve {int(100 * fraction)}%")
            match = re.search(r"\bcycle\s+(\d+)/(\d+)", line)
            if match and int(match.group(2)):
                fraction = int(match.group(1)) / int(match.group(2))
                self._update_progress(
                    step, 71 + 12 * fraction, "running",
                    f"Surface recoupling {int(100 * fraction)}%")
            match = re.search(r"\bstep\s+(\d+)/(\d+)", line)
            if match and int(match.group(2)) and "microclimate step" not in line:
                fraction = int(match.group(1)) / int(match.group(2))
                self._update_progress(
                    step, 83 + 15 * fraction, "running",
                    f"MRT regeneration {int(100 * fraction)}%")

        if step == 5:
            phase_markers = (
                ("Loading results...", 5, "Loading MRT results"),
                ("Building static key-times overview", 18, "Drawing MRT overview"),
                ("Building interactive animated HTML (subsampled", 30, "Writing MRT animation"),
                ("Computing UTCI for every point", 56, "Computing UTCI field"),
                ("Building static overview with standard UTCI", 72, "Drawing UTCI overview"),
                ("Building interactive animated HTML", 86, "Writing UTCI animation"),
            )
            for token, percent, message in phase_markers:
                if token in line:
                    self._update_progress(step, percent, "running", message)
                    break

        if step == 6:
            loaded = re.search(r"Loaded\s+(\d+)\s+routes", line)
            if loaded:
                with self.lock:
                    self.route_count = max(1, int(loaded.group(1)))
            if "Computing UTCI along each route" in line:
                with self.lock:
                    self.route_phase = "utci"
                self._update_progress(step, 8, "running", "Computing route UTCI")
            elif "Simulating the walk along each route with JOS-3" in line:
                with self.lock:
                    self.route_phase = "jos3"
                self._update_progress(step, 58, "running", "Simulating JOS-3 routes")
            route = re.search(r"^Route\s+(\d+):", line.strip())
            if route:
                with self.lock:
                    count, phase = max(1, self.route_count), self.route_phase
                fraction = min(int(route.group(1)), count) / count
                if phase == "utci":
                    self._update_progress(
                        step, 8 + 38 * fraction, "running",
                        f"Route UTCI {int(100 * fraction)}%")
                elif phase == "jos3":
                    self._update_progress(
                        step, 58 + 37 * fraction, "running",
                        f"JOS-3 routes {int(100 * fraction)}%")
        return False

    def _pump(self):
        for line in self.proc.stdout:
            clean = line.rstrip("\n")
            if not self._parse_progress(clean):
                self._append(clean)
        code = self.proc.wait()
        with self.lock:
            if self.status == "stopping":
                self.status = "stopped"
                self.lines.append("\n--- stopped by user ---")
                if self.active_step in self.planned_steps:
                    self.progress[self.active_step]["state"] = "stopped"
                    self.progress[self.active_step]["message"] = "Stopped by user"
            elif code == 0:
                self.status = "done"
                self.lines.append("\n--- finished successfully ---")
                for step in self.planned_steps:
                    item = self.progress[step]
                    if item["state"] == "queued":
                        item.update(percent=100.0, state="skipped",
                                    message="Skipped by configuration")
                    elif item["state"] == "running":
                        item.update(percent=100.0, state="done", message="Complete")
            else:
                self.status = "failed"
                self.lines.append(f"\n--- exited with code {code} ---")
                if self.active_step in self.planned_steps:
                    self.progress[self.active_step]["state"] = "failed"
                    self.progress[self.active_step]["message"] = f"Failed (exit {code})"

    def stop(self):
        if not self.is_running():
            return False
        with self.lock:
            self.status = "stopping"
            if self.active_step in self.planned_steps:
                self.progress[self.active_step]["state"] = "stopping"
                self.progress[self.active_step]["message"] = "Stopping"
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            self.proc.terminate()
        return True


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>TREC-Route Pipeline</title>
<style>
 body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#11151a;color:#dfe6ee}
 header{padding:14px 20px;background:#171d24;border-bottom:1px solid #263140;
        display:flex;align-items:center;gap:14px;flex-wrap:wrap}
 h1{font-size:16px;margin:0;font-weight:600}
 #status{font-size:13px;padding:3px 10px;border-radius:20px;background:#263140}
 #status.idle{color:#9fb0c3}#status.running{background:#1d4ed8!important;color:#fff}
 #status.done{background:#15803d!important;color:#fff}
 #status.failed{background:#b91c1c!important;color:#fff}
 #status.stopped,#status.stopping{background:#a16207!important;color:#fff}
 main{padding:16px 20px}
 table{border-collapse:collapse;width:100%;max-width:1220px}
 td{padding:5px 8px 5px 0;vertical-align:middle}
 td.n{color:#7d8ea3;width:26px}
 td.l{width:100%}
 td.p{width:285px;padding-left:12px}
 button{font:13px system-ui,sans-serif;padding:6px 12px;border-radius:6px;
        border:1px solid #33465c;background:#1e2833;color:#dfe6ee;cursor:pointer;
        white-space:nowrap}
 button:hover:not(:disabled){background:#27384a}
 button:disabled{opacity:.4;cursor:not-allowed}
 button.go{border-color:#2b6cb0;background:#1c3f66}
button.stop{border-color:#b91c1c;background:#5b1414}
 .case-panel{max-width:1220px;margin:0 0 14px;padding:12px 14px;border-radius:8px;
             border:1px solid #33465c;background:#171d24;display:flex;
             align-items:center;gap:12px;flex-wrap:wrap}
 .case-number{color:#93c5fd;font-weight:700}.case-title{font-weight:600;margin-right:8px}
 .case-panel label{color:#9fb0c3;font-size:12px}
 select{margin-left:5px;padding:6px 28px 6px 9px;border-radius:6px;
        border:1px solid #33465c;background:#0e1319;color:#dfe6ee}
 .case-path{color:#7d8ea3;font:11px ui-monospace,monospace;flex-basis:100%}
 .progress-wrap{display:grid;grid-template-columns:minmax(150px,1fr) 104px;
                gap:8px;align-items:center;min-width:265px}
 .progress-track{height:12px;border-radius:999px;overflow:hidden;
                 background:#0c1117;border:1px solid #304055}
 .progress-fill{height:100%;width:0%;background:#64748b;
                transition:width .35s ease,background-color .2s ease}
 .progress-wrap.running .progress-fill{background:#3b82f6}
 .progress-wrap.queued .progress-fill{background:#475569}
 .progress-wrap.done .progress-fill{background:#22c55e}
 .progress-wrap.failed .progress-fill{background:#ef4444}
 .progress-wrap.stopped .progress-fill,.progress-wrap.stopping .progress-fill{background:#f59e0b}
 .progress-wrap.skipped .progress-fill{background:#64748b}
 .progress-text{font:11px/1.2 system-ui,sans-serif;color:#9fb0c3;
                white-space:nowrap;text-align:left}
 #envrow{margin:14px 0 6px;max-width:1220px}
 #env{width:100%;padding:7px 9px;border-radius:6px;border:1px solid #33465c;
      background:#0e1319;color:#dfe6ee;font:13px ui-monospace,monospace}
 #hint{color:#7d8ea3;font-size:12px;margin-top:5px}
 #log{margin-top:14px;background:#0b0f14;border:1px solid #263140;border-radius:8px;
      padding:12px;height:52vh;overflow:auto;white-space:pre-wrap;
      font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#cbd5e1}
</style></head><body>
<header>
  <h1>TREC-Route Pipeline</h1>
  <span id="status" class="idle">idle</span>
  <span id="current" style="color:#7d8ea3;font-size:13px"></span>
  <button class="stop" id="stopb" onclick="stop()" disabled>Stop</button>
</header>
<main>
 <div class="case-panel">
  <span class="case-number">1</span><span class="case-title">Problem selection</span>
  <label>Input case <select id="inputcase"></select></label>
  <label>Output case <select id="outputcase"></select></label>
  <span class="case-path" id="casepath">Loading available cases…</span>
 </div>
 <div id="envrow">
  <input id="env" placeholder="optional overrides, e.g.  DEPARTURE_HOUR=15.0 SUBJECT_PROFILE=child">
  <div id="hint">Left button = run only that step. Right button = that step and everything after. Progress is measured from completed pipeline phases and numerical loop counters.</div>
 </div>
 <table id="steps"></table>
 <div id="log"></div>
</main>
<script>
const STEPS=__STEPS__;
const t=document.getElementById('steps');
STEPS.forEach(([n,label])=>{
  const tr=document.createElement('tr');
  tr.innerHTML=`<td class="n">${n}</td><td class="l">${label}</td>
    <td><button onclick="run(${n},true)">Only this</button></td>
    <td><button class="go" onclick="run(${n},false)">This + after</button></td>
    <td class="p"><div class="progress-wrap idle" id="progress-${n}"
      role="progressbar" aria-label="${label} progress" aria-valuemin="0"
      aria-valuemax="100" aria-valuenow="0">
      <div class="progress-track"><div class="progress-fill"></div></div>
      <span class="progress-text">not run</span></div></td>`;
  t.appendChild(tr);
});
let offset=0, busy=false;
const logEl=document.getElementById('log'), stEl=document.getElementById('status'),
      curEl=document.getElementById('current'), stopB=document.getElementById('stopb');
async function run(step,only){
  if(busy) return;
  logEl.textContent=''; offset=0;
  const response=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({step:step,only:only,env:document.getElementById('env').value,
      input_case:document.getElementById('inputcase').value,
      output_case:document.getElementById('outputcase').value})});
  const result=await response.json();
  if(!result.ok){ logEl.textContent='ERROR: '+result.msg+'\\n'; return; }
  poll();
}
async function stop(){ await fetch('/stop',{method:'POST'}); }
function setBusy(b){
  busy=b; stopB.disabled=!b;
  document.querySelectorAll('#steps button').forEach(x=>x.disabled=b);
  document.getElementById('inputcase').disabled=b;
  document.getElementById('outputcase').disabled=b;
}
function updateCasePath(){
  const input=document.getElementById('inputcase').value;
  const output=document.getElementById('outputcase').value;
  document.getElementById('casepath').textContent=`input/${input}  →  run_output/${output}`;
}
async function loadCases(){
  const response=await fetch('/cases'); const data=await response.json();
  const input=document.getElementById('inputcase'), output=document.getElementById('outputcase');
  input.innerHTML=''; output.innerHTML='';
  (data.input_cases||[]).forEach(name=>input.add(new Option(name,name)));
  (data.output_cases||[]).forEach(name=>output.add(new Option(name,name)));
  const preferred=(data.input_cases||[]).includes('MMC')?'MMC':(data.input_cases||[])[0];
  if(preferred){ input.value=preferred; output.value=preferred; }
  input.onchange=()=>{
    if([...output.options].some(option=>option.value===input.value)) output.value=input.value;
    updateCasePath();
  };
  output.onchange=updateCasePath; updateCasePath();
}
function updateProgress(progress){
  STEPS.forEach(([n])=>{
    const item=(progress||{})[String(n)]||{percent:0,state:'idle',message:'Not run'};
    const wrap=document.getElementById('progress-'+n);
    const percent=Math.max(0,Math.min(100,Number(item.percent)||0));
    wrap.className='progress-wrap '+item.state;
    wrap.querySelector('.progress-fill').style.width=percent+'%';
    wrap.setAttribute('aria-valuenow',percent.toFixed(1));
    let label=item.state==='idle'?'not run':item.state==='queued'?'waiting':
      item.state==='skipped'?'skipped':item.state==='failed'?`failed ${Math.round(percent)}%`:
      item.state==='stopped'?`stopped ${Math.round(percent)}%`:
      `${Math.round(percent)}%`;
    if(item.message && !['idle','queued'].includes(item.state)) label+=' · '+item.message;
    wrap.querySelector('.progress-text').textContent=label;
    wrap.title=item.message||'';
  });
}
async function poll(){
  const r=await fetch('/log?offset='+offset);
  const d=await r.json();
  if(d.lines.length){
    const atBottom = logEl.scrollHeight-logEl.scrollTop-logEl.clientHeight < 40;
    logEl.textContent += d.lines.join('\\n')+'\\n';
    offset=d.offset;
    if(atBottom) logEl.scrollTop=logEl.scrollHeight;
  }
  stEl.textContent=d.status; stEl.className=d.status;
  curEl.textContent=d.current||'';
  updateProgress(d.progress);
  setBusy(d.status==='running'||d.status==='stopping');
  if(d.status==='running'||d.status==='stopping') setTimeout(poll,600);
}
loadCases(); poll();
</script></body></html>"""


def make_handler(runner):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # keep the console clean; the browser shows everything

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                steps = json.dumps([[n, html.escape(l)] for n, l, _ in STEPS])
                self._send(200, PAGE.replace("__STEPS__", steps), "text/html; charset=utf-8")
            elif self.path == "/cases":
                input_cases, output_cases = runner.available_cases()
                self._send(200, json.dumps({"input_cases": input_cases,
                                            "output_cases": output_cases}))
            elif self.path.startswith("/log"):
                q = self.path.split("?", 1)[1] if "?" in self.path else ""
                off = 0
                for part in q.split("&"):
                    if part.startswith("offset="):
                        off = int(part[7:] or 0)
                lines, total, status, current, progress = runner.snapshot(off)
                self._send(200, json.dumps({"lines": lines, "offset": total,
                                            "status": status, "current": current,
                                            "progress": progress}))
            else:
                self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or "{}") if n else {}
            if self.path == "/run":
                ok, msg = runner.start(int(payload.get("step", 1)),
                                       bool(payload.get("only", False)),
                                       payload.get("env", ""),
                                       payload.get("input_case", "MMC"),
                                       payload.get("output_case", "MMC"))
                self._send(200, json.dumps({"ok": ok, "msg": msg}))
            elif self.path == "/stop":
                self._send(200, json.dumps({"ok": runner.stop()}))
            else:
                self._send(404, "{}")
    return H


def main():
    ap = argparse.ArgumentParser(description="TREC-Route Pipeline browser UI")
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Directory containing start.sh (default: this file's dir)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    workdir = os.path.abspath(args.dir)
    script = os.path.join(workdir, "start.sh")
    if not os.path.isfile(script):
        raise SystemExit(f"start.sh not found in {workdir}\n"
                         f"Run from the pipeline folder or pass --dir")

    runner = Runner(workdir)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runner))
    url = f"http://127.0.0.1:{args.port}"
    print(f"TREC-Route Pipeline UI  ->  {url}")
    print(f"  working directory: {workdir}")
    print("  (Ctrl-C to quit)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    main()
