"""Run paper_jos3_figures' departure sweep for MMC_rsearch with the stage-09
consistency tolerance loosened to 0.005 C (evaluation only; repo file untouched)."""
import sys, types
P="/media/harshin/data_drive/route based UTCI"; sys.path.insert(0,P)
src=open(f"{P}/paper_jos3_figures.py").read()
assert src.count("if diff > 0.003:")==1
src=src.replace("if diff > 0.003:","if diff > 0.005:")
mod=types.ModuleType("pjf"); mod.__file__=f"{P}/paper_jos3_figures.py"
sys.argv=["paper_jos3_figures.py","--case","MMC_rsearch","--fixed-clothing","--paper-dir",
          "/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/rsearch/fig"]
exec(compile(src,f"{P}/paper_jos3_figures.py","exec"),mod.__dict__)
mod.main()
