#!/usr/bin/env python3
"""Generate the TREC-Route implementation and physics book as a PDF.

The output is a self-contained technical description of Thermal Risk and
Exposure Computation for Routes (TREC-Route): what the software calculates,
how data move through the case-driven workflow, the governing radiation and
surface-energy equations, route-time integration, UTCI and JOS-3 coupling,
diagnostic outputs, verification, assumptions, and limitations.

The document is generated from vector text, tables, and diagrams with
Matplotlib.  It does not require LaTeX, ReportLab, a browser, or Internet
access.  Selected material defaults are read from the current centralized
material database so the book does not silently drift from the implementation.

Examples
--------
Generate the standard generic book in the project folder::

    python3 generate_book.py

Choose the output file and document metadata::

    python3 generate_book.py --output documentation/TREC-Route_book.pdf \
        --author "Research Group" --version "1.0"

Append a page describing one concrete case manifest::

    python3 generate_book.py --case input/MMC
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import textwrap
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge
import numpy as np


ROOT = Path(__file__).resolve().parent
A4 = (8.27, 11.69)
NAVY = "#12324A"
TEAL = "#007C83"
CYAN = "#51B6C8"
ORANGE = "#E07A2D"
RED = "#B33A3A"
GOLD = "#D7A928"
INK = "#202A33"
MUTED = "#5C6973"
PALE = "#EDF5F5"
PALE_BLUE = "#ECF3F8"
PALE_ORANGE = "#FAF0E8"


@dataclass
class Page:
    chapter: str
    title: str
    renderer: Callable[[plt.Figure, "BookContext"], None]


@dataclass
class BookContext:
    author: str
    version: str
    generated: str
    total_pages: int
    page_number: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the TREC-Route implementation and physics PDF book")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "TREC-Route_Implementation_and_Physics.pdf")
    parser.add_argument("--author", default="TREC-Route project")
    parser.add_argument("--version", default="current local implementation")
    parser.add_argument(
        "--case", type=Path, default=None,
        help="Optional input case directory (or case.json) summarized in an appendix")
    parser.add_argument(
        "--title", default="TREC-Route",
        help="Cover-page software title")
    return parser.parse_args()


def wrap(text: str, width: int = 94) -> list[str]:
    return textwrap.wrap(
        " ".join(str(text).split()), width=width,
        break_long_words=False, break_on_hyphens=False) or [""]


def page_frame(fig: plt.Figure, page: Page, ctx: BookContext) -> None:
    fig.patch.set_facecolor("white")
    fig.text(0.065, 0.958, page.chapter.upper(), fontsize=8.2,
             fontweight="bold", color=TEAL, va="top")
    fig.text(0.065, 0.925, page.title, fontsize=18, fontweight="bold",
             color=NAVY, va="top")
    fig.add_artist(plt.Line2D([0.065, 0.935], [0.885, 0.885],
                              transform=fig.transFigure, color=CYAN, lw=1.1))
    fig.text(0.065, 0.027, "TREC-Route · implementation and physics",
             fontsize=7.2, color=MUTED)
    fig.text(0.935, 0.027, f"{ctx.page_number} / {ctx.total_pages}",
             fontsize=7.2, color=MUTED, ha="right")


def body_text(fig: plt.Figure, text: str, x: float, y: float,
              width: int = 94, size: float = 9.25,
              color: str = INK, line_height: float = 0.0205,
              weight: str = "normal") -> float:
    lines = wrap(text, width)
    fig.text(x, y, "\n".join(lines), fontsize=size, color=color,
             va="top", linespacing=1.38, fontweight=weight)
    return y - line_height * len(lines)


def draw_blocks(fig: plt.Figure, blocks: list[dict], start_y: float = 0.852) -> None:
    y = start_y
    for block in blocks:
        heading = block.get("heading")
        if heading:
            fig.text(0.075, y, heading, fontsize=11.2, fontweight="bold",
                     color=NAVY, va="top")
            y -= 0.033
        if block.get("formula"):
            formula = block["formula"]
            # Matplotlib only enters its built-in mathtext renderer inside
            # dollar delimiters.  Keep callers readable by accepting bare
            # TeX-like expressions and add the delimiters here once.
            if not (formula.startswith("$") and formula.endswith("$")):
                formula = f"${formula}$"
            fig.text(0.095, y, formula, fontsize=12.0, color=TEAL, va="top")
            y -= 0.048
        if block.get("text"):
            y = body_text(fig, block["text"], 0.085, y,
                          width=block.get("width", 94),
                          size=block.get("size", 9.25)) - 0.018
        for bullet in block.get("bullets", []):
            lines = wrap(bullet, 87)
            fig.text(0.093, y, "•", fontsize=10, color=ORANGE, va="top")
            fig.text(0.113, y, "\n".join(lines), fontsize=9.1, color=INK,
                     va="top", linespacing=1.35)
            y -= 0.0205 * len(lines) + 0.010
        if block.get("note"):
            note_lines = wrap(block["note"], 84)
            height = 0.026 + len(note_lines) * 0.020
            fig.add_artist(Rectangle((0.075, y - height + 0.006), 0.85, height,
                                     transform=fig.transFigure, facecolor=PALE_ORANGE,
                                     edgecolor="#E8C7AA", lw=0.8))
            fig.text(0.092, y - 0.007, "\n".join(note_lines), fontsize=8.7,
                     color="#6D4426", va="top", linespacing=1.35)
            y -= height + 0.018
    if y < 0.065:
        raise RuntimeError("book page overflow; shorten the page content")


def text_page(chapter: str, title: str, blocks: list[dict]) -> Page:
    def render(fig: plt.Figure, ctx: BookContext) -> None:
        draw_blocks(fig, blocks)
    return Page(chapter, title, render)


def diagram_page(chapter: str, title: str,
                 drawer: Callable[[plt.Figure, BookContext], None]) -> Page:
    return Page(chapter, title, drawer)


def cover(fig: plt.Figure, ctx: BookContext, title: str) -> None:
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for radius, alpha in ((0.40, 0.06), (0.30, 0.09), (0.20, 0.12)):
        ax.add_patch(Circle((0.82, 0.80), radius, color=CYAN, alpha=alpha))
    route = np.array([[0.08, 0.23], [0.20, 0.31], [0.31, 0.27], [0.43, 0.40],
                      [0.57, 0.36], [0.69, 0.49], [0.86, 0.45]])
    ax.plot(route[:, 0], route[:, 1], color=CYAN, lw=5, solid_capstyle="round")
    ax.scatter(route[:, 0], route[:, 1], s=45, color="white", zorder=4)
    for x, h, c in ((0.18, .17, "#446476"), (.39, .24, "#355668"),
                    (.65, .19, "#446476"), (.78, .29, "#355668")):
        ax.add_patch(Rectangle((x, .50), .09, h, fc=c, ec="none"))
    ax.add_patch(Wedge((0.16, .78), .105, 0, 360, fc="#4A8D68", ec="none"))
    ax.add_patch(Wedge((0.27, .72), .08, 0, 360, fc="#5B9C75", ec="none"))
    fig.text(0.08, 0.88, title, fontsize=32, fontweight="bold", color="white", va="top")
    fig.text(0.08, 0.815, "Thermal Risk and Exposure Computation for Routes",
             fontsize=14, color="#BFE7EA", va="top")
    fig.text(0.08, 0.745, "Implementation and Physics", fontsize=23,
             fontweight="bold", color="#F4B77E", va="top")
    fig.text(0.08, 0.665,
             "A technical guide to route-resolved urban radiation, surface energy,\n"
             "mean radiant temperature, UTCI exposure, and JOS-3 heat strain",
             fontsize=11.5, color="white", va="top", linespacing=1.5)
    fig.text(0.08, 0.115, f"Version: {ctx.version}\nGenerated: {ctx.generated}\n{ctx.author}",
             fontsize=9.5, color="#D8E5EA", va="bottom", linespacing=1.55)


def contents(fig: plt.Figure, ctx: BookContext, entries: list[tuple[str, int]],
             part: int, total_parts: int) -> None:
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.94, "CONTENTS", fontsize=9, fontweight="bold", color=TEAL)
    fig.text(0.07, 0.895, "Table of contents", fontsize=24, fontweight="bold", color=NAVY)
    selected = entries[(part - 1) * 12: part * 12]
    y = 0.83
    for index, (label, page_no) in enumerate(selected, 1 + (part - 1) * 12):
        fig.text(0.08, y, f"{index:02d}", fontsize=9, fontweight="bold", color=ORANGE)
        fig.text(0.13, y, label, fontsize=10.3, color=INK)
        fig.add_artist(plt.Line2D([0.13, 0.86], [y - .006, y - .006],
                                  transform=fig.transFigure, color="#D8E2E6",
                                  lw=.55, ls=(0, (1.2, 2.2))))
        fig.text(0.91, y, str(page_no), fontsize=10, color=NAVY, ha="right")
        y -= 0.061
    fig.text(0.07, 0.035, f"Contents {part} of {total_parts}", fontsize=7.5, color=MUTED)


def workflow_diagram(fig: plt.Figure, ctx: BookContext) -> None:
    ax = fig.add_axes([0.07, 0.14, 0.86, 0.69]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    boxes = [
        (0.2, 8.2, 2.2, 1.0, "Case selection", "input/<case>\nrun_output/<case>"),
        (2.8, 8.2, 2.2, 1.0, "OSM materials", "full tags → terrain IDs"),
        (5.4, 8.2, 2.2, 1.0, "MRT preparation", "route points · time · SVF"),
        (7.9, 8.2, 1.9, 1.0, "Facet selection", "visible surfaces"),
        (7.9, 5.9, 1.9, 1.0, "Energy balance", "surface T + radiosity"),
        (5.4, 5.9, 2.2, 1.0, "Final MRT", "SW + LW absorbed flux"),
        (2.8, 5.9, 2.2, 1.0, "Route sampling", "arrival-time environment"),
        (0.2, 5.9, 2.2, 1.0, "Thermal outcomes", "UTCI · JOS-3 · ranking"),
    ]
    for x, y, w, h, title, sub in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fc=PALE_BLUE, ec=TEAL, lw=1.2))
        ax.text(x + w/2, y + .66, title, ha="center", va="center", fontsize=9.3,
                fontweight="bold", color=NAVY)
        ax.text(x + w/2, y + .26, sub, ha="center", va="center", fontsize=7.6, color=MUTED)
    sequence = [(2.4, 8.7, 2.8, 8.7), (5.0, 8.7, 5.4, 8.7),
                (7.6, 8.7, 7.9, 8.7), (8.85, 8.2, 8.85, 6.9),
                (7.9, 6.4, 7.6, 6.4), (5.4, 6.4, 5.0, 6.4),
                (2.8, 6.4, 2.4, 6.4)]
    for x1, y1, x2, y2 in sequence:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.5, color=ORANGE))
    ax.add_patch(Rectangle((0.4, 2.6), 4.0, 1.35, fc=PALE_ORANGE, ec=ORANGE, lw=1.0))
    ax.text(2.4, 3.45, "Preparation outside the general UI", ha="center",
            fontsize=10, fontweight="bold", color="#7A4522")
    ax.text(2.4, 3.05, "geometry construction · route generation/import",
            ha="center", fontsize=8.3, color="#7A4522")
    ax.add_patch(Rectangle((5.6, 2.6), 4.0, 1.35, fc=PALE, ec=TEAL, lw=1.0))
    ax.text(7.6, 3.45, "Authoritative route order is preserved", ha="center",
            fontsize=10, fontweight="bold", color=NAVY)
    ax.text(7.6, 3.05, "sorting is allowed only in diagnostic plot copies",
            ha="center", fontsize=8.3, color=MUTED)
    fig.text(0.08, 0.105,
             "The numbered workflow begins after a case already contains geometry and routes. "
             "OSM routing and OSM physical surfaces are deliberately separate representations.",
             fontsize=9.4, color=INK)


def ray_diagram(fig: plt.Figure, ctx: BookContext) -> None:
    ax = fig.add_axes([0.07, 0.18, 0.86, 0.65]); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 10, .42, fc="#B9A27B", ec="#78684B"))
    ax.add_patch(Rectangle((7.7, .42), 1.4, 4.3, fc="#9AA7AD", ec="#55656D"))
    for x, r in ((2.0, .85), (3.0, .65)):
        ax.add_patch(Circle((x, 3.0), r, fc="#4F9368", ec="#306348"))
        ax.plot([x, x], [.42, 2.3], color="#765238", lw=4)
    ax.add_patch(Circle((5.2, 1.65), .27, fc=INK)); ax.plot([5.2,5.2],[1.38,.72],color=INK,lw=4)
    ax.plot([5.2,4.8],[1.08,.52],color=INK,lw=2.5); ax.plot([5.2,5.6],[1.08,.52],color=INK,lw=2.5)
    rays = [((5.2,1.75),(1.0,6.2),"sky",CYAN), ((5.2,1.75),(2.2,3.3),"canopy",TEAL),
            ((5.2,1.75),(8.0,3.5),"wall",ORANGE), ((5.2,1.55),(6.8,.42),"ground",GOLD)]
    for start,end,label,color in rays:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=12,
                                     color=color,lw=1.8))
        ax.text(end[0],end[1]+.18,label,ha="center",fontsize=8,color=color)
    ax.add_patch(FancyArrowPatch((.7,6.5),(5.0,2.0),arrowstyle="-|>",mutation_scale=14,
                                 color="#F5A623",lw=2.5))
    ax.text(.9,6.55,"direct sun vector",fontsize=9,color="#A86600")
    fig.text(0.08, 0.135,
             "Solid building/ground intersections block rays. Vegetation intersections attenuate "
             "radiation with Beer–Lambert transmission. First-hit IDs retain the source material.",
             fontsize=9.3, color=INK)


def energy_diagram(fig: plt.Figure, ctx: BookContext) -> None:
    ax = fig.add_axes([0.08, 0.19, 0.84, 0.63]); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.add_patch(Rectangle((3.3, 3.0), 3.4, .38, fc="#8A6D4A", ec="#654C31"))
    for i, color in enumerate(["#C69A65", "#B98A55", "#A77A48", "#927044", "#7C653F"]):
        ax.add_patch(Rectangle((3.3, 2.35-i*.34), 3.4, .32, fc=color, ec="white", lw=.4))
    ax.text(5.0, 1.0, "graded one-dimensional substrate", ha="center", fontsize=9, color=MUTED)
    items = [
        ((1.0,6.0),(3.7,3.5),"absorbed shortwave",ORANGE),
        ((4.2,6.2),(4.8,3.5),"absorbed incident LW",TEAL),
        ((6.0,3.5),(8.7,5.7),"emitted LW  εσTₛ⁴",RED),
        ((6.3,3.25),(8.8,3.25),"convection  h(Tₛ−Tₐ)",NAVY),
        ((4.9,2.95),(4.9,.55),"conduction",GOLD),
        ((3.5,3.05),(1.0,1.6),"latent heat where enabled",CYAN),
    ]
    for start,end,label,color in items:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=13,
                                     lw=2,color=color))
        ax.text(start[0],start[1]+.2,label,fontsize=8.5,color=color,ha="center")
    fig.text(0.08, 0.135,
             "Each route-visible facet uses the material’s albedo, emissivity, conductivity, "
             "volumetric heat capacity, effective depth, boundary condition, and evaporative efficiency.",
             fontsize=9.2, color=INK)


def radiosity_diagram(fig: plt.Figure, ctx: BookContext) -> None:
    ax = fig.add_axes([0.08, 0.19, 0.84, 0.62]); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    positions = [(2,4.8),(5,5.7),(8,4.5),(6.8,1.7),(3.0,1.6)]
    colors = ["#8E9BA1","#B7A18A","#567E61","#6A7480","#B58B58"]
    for i,((x,y),c) in enumerate(zip(positions,colors)):
        ax.add_patch(Circle((x,y),.58,fc=c,ec=NAVY,lw=1.1))
        ax.text(x,y,f"facet {i+1}\nεᵢ, Tᵢ",ha="center",va="center",fontsize=8,color="white")
    for i in range(len(positions)):
        x1,y1=positions[i]; x2,y2=positions[(i+1)%len(positions)]
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=10,
                                     lw=1,color=TEAL,alpha=.75,connectionstyle="arc3,rad=.12"))
    ax.add_patch(Wedge((5,8.0),4.7,200,340,fc=PALE_BLUE,ec=CYAN,lw=1.0))
    ax.text(5,6.75,"sky boundary  Lsky",ha="center",fontsize=10,color=TEAL,fontweight="bold")
    ax.text(5,3.55,"route-visible enclosure\narea/non-sky weighted J̄",ha="center",
            fontsize=11,color=NAVY,fontweight="bold")
    fig.text(0.08, 0.135,
             "The mean-field grey enclosure preserves emission plus reflected incident longwave and "
             "closes every facet to the sky. It avoids claiming unavailable full facet-to-facet view factors.",
             fontsize=9.2, color=INK)


def route_timeline(fig: plt.Figure, ctx: BookContext) -> None:
    ax = fig.add_axes([0.08, 0.21, 0.84, 0.58]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    x = np.linspace(.7,9.3,12); y=3.1+.55*np.sin(np.linspace(0,2.7*np.pi,12))
    ax.plot(x,y,color=TEAL,lw=3)
    exposures=["shade","sun","trees","sun","wall","shade","road","sun","plaza","trees","shade","finish"]
    for i,(xx,yy,label) in enumerate(zip(x,y,exposures)):
        col=ORANGE if label=="sun" else ("#4F9368" if label=="trees" else NAVY)
        ax.scatter(xx,yy,s=55,color=col,zorder=3)
        ax.text(xx,yy+.45,label,ha="center",fontsize=7.5,color=col)
        ax.text(xx,yy-.5,f"t{i}",ha="center",fontsize=7,color=MUTED)
    ax.add_patch(FancyArrowPatch((.7,1.2),(9.3,1.2),arrowstyle="-|>",mutation_scale=12,lw=1.5,color=INK))
    ax.text(5,0.75,"recorded time or cumulative distance / walking speed",ha="center",fontsize=9,color=INK)
    fig.text(0.08,0.135,
             "MRT and weather are sampled at each point’s arrival time. UTCI is evaluated pointwise; "
             "JOS-3 advances one physiological state sequentially through the same timeline.",
             fontsize=9.3,color=INK)


def validation_diagram(fig: plt.Figure, ctx: BookContext) -> None:
    ax=fig.add_axes([.10,.18,.80,.65]); ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis("off")
    levels=[("Analytic identities","flux and radiosity conservation",1.0,1.3,TEAL),
            ("Synthetic scenes","known shade, materials and route order",1.6,2.5,CYAN),
            ("Regression checks","disabled-mode and cache equivalence",2.2,3.7,GOLD),
            ("Independent models","SOLWEIG and ENVI-met comparisons",2.8,4.9,ORANGE),
            ("Field observations","globe/radiometer/mobile measurements",3.4,6.1,RED)]
    for title,sub,half,y,c in levels:
        ax.add_patch(Polygon([[5-half,y-.48],[5+half,y-.48],[5+half-.35,y+.48],[5-half+.35,y+.48]],
                             closed=True,fc=c,ec="white",alpha=.88))
        ax.text(5,y+.08,title,ha="center",fontsize=9,fontweight="bold",color="white")
        ax.text(5,y-.22,sub,ha="center",fontsize=7.3,color="white")
    fig.text(.08,.12,
             "Agreement is reported without tuning physical parameters to the validation observations. "
             "Disagreement is decomposed into forcing, geometry/shade, surface temperature, and sensor equivalence.",
             fontsize=9.3,color=INK)


def material_table_page(materials: dict) -> Page:
    preferred = ["generic_ground", "asphalt_road", "asphalt_pedestrian",
                 "concrete_pedestrian", "paving_stone_pedestrian",
                 "asphalt_parking", "grass_lawn", "bare_ground", "water"]
    rows=[]
    for name in preferred:
        if name not in materials:
            continue
        m=materials[name]
        rows.append([name, f"{float(m['albedo']):.2f}", f"{float(m['emissivity']):.2f}",
                     f"{float(m['k']):.2f}", f"{float(m['C'])/1e6:.2f}",
                     f"{float(m['depth']):.2f}",
                     f"{float(m.get('evaporative_efficiency',0)):.2f}"])
    def render(fig: plt.Figure, ctx: BookContext) -> None:
        ax=fig.add_axes([.065,.25,.87,.57]); ax.axis("off")
        table=ax.table(cellText=rows,
            colLabels=["Material","albedo","emiss.","k\nW m⁻¹ K⁻¹","C\nMJ m⁻³ K⁻¹","depth\nm","evap.\neff."],
            cellLoc="center",colLoc="center",loc="upper center",
            colWidths=[.25,.10,.10,.13,.15,.10,.10])
        table.auto_set_font_size(False); table.set_fontsize(7.7); table.scale(1,1.65)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor("white")
            if r==0:
                cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(PALE_BLUE if r%2 else "#F7FAFC")
        fig.text(.075,.205,
            "Values are current configurable defaults read from osm_ground_materials.py. They are typical "
            "assumptions, not universal constants or site measurements. C is volumetric heat capacity; "
            "evaporative efficiency is the first-order water-availability factor used by the optional "
            "equilibrium latent-heat term.",fontsize=9,color=INK,va="top",wrap=True)
    return Page("Chapter 3 · ground materials", "Selected default material properties", render)


def case_page(case_path: Path) -> Page:
    path = case_path / "case.json" if case_path.is_dir() else case_path
    data = json.loads(path.read_text(encoding="utf-8"))
    loc=data.get("location",{}); coord=data.get("coordinates",{}); defaults=data.get("simulation_defaults",{})
    blocks=[
        {"heading":"Manifest", "text":f"Case ID: {data.get('case_id')}. Site: {data.get('site_name')}. Source manifest: {path.resolve()}."},
        {"heading":"Location and frame", "bullets":[
            f"Latitude/longitude: {loc.get('latitude')}, {loc.get('longitude')}; timezone: {loc.get('timezone')}",
            f"Projected CRS: {coord.get('project_crs')}; local origin: ({coord.get('local_origin_x')}, {coord.get('local_origin_y')})",
            f"OSM bounds: {loc.get('osm_bbox_wgs84')}"]},
        {"heading":"Default run", "bullets":[
            f"Date: {defaults.get('date')}; departure hour: {defaults.get('departure_hour')}",
            f"Timing: {defaults.get('timing_mode')}; fallback walking speed: {defaults.get('walking_speed_ms')} m s⁻¹",
            f"Default RH/wind/cloud: {defaults.get('relative_humidity_pct')} %, {defaults.get('wind_speed_ms')} m s⁻¹, {defaults.get('cloud_cover_fraction')}"]},
        {"heading":"Interpretation", "note":"This appendix records configuration, not results. The generic methods in the preceding chapters are unchanged by case selection."}
    ]
    return text_page("Appendix · selected case", f"Case manifest: {data.get('case_id')}", blocks)


def load_materials() -> dict:
    try:
        from osm_ground_materials import DEFAULT_CONFIG
        return DEFAULT_CONFIG["materials"]
    except Exception as exc:
        print(f"WARNING: could not import live material database ({exc}); omitting table")
        return {}


def build_pages(materials: dict, selected_case: Path | None) -> list[Page]:
    pages: list[Page] = []
    pages += [
        text_page("Chapter 1 · purpose", "What TREC-Route calculates", [
            {"heading":"Research objective", "text":"TREC-Route evaluates outdoor thermal exposure along pedestrian routes while retaining the order and time at which a walker encounters shade, sun, wind and urban surfaces. It connects three-dimensional scene physics to route-scale comfort and physiological heat strain rather than reducing a route to one location or one steady environment."},
            {"heading":"Primary products", "bullets":["Time- and receptor-resolved mean radiant temperature (MRT).", "Material-resolved absorbed shortwave and longwave flux in W m⁻².", "UTCI along each route, route means, peaks and strong-heat-stress dose.", "Sequential JOS-3 body temperatures and final core-temperature rise.", "Maps, animations, route exports, rankings and validation diagnostics."]},
            {"heading":"Two questions kept separate", "text":"Route-selection fidelity asks which route is thermally preferable. Absolute-prediction accuracy asks whether the numerical MRT, UTCI or core-temperature rise is correct. A method may preserve route ordering while retaining a common bias; the software reports both outcomes where the analysis supports them."},
        ]),
        text_page("Chapter 1 · purpose", "Scope and scientific boundaries", [
            {"heading":"What is inside the model", "text":"The active workflow covers case selection, complete OSM surface classification, route-local thermal facets, solar and longwave radiation, material-specific surface energy, absorbed pedestrian flux, MRT, UTCI and JOS-3 route integration."},
            {"heading":"What remains case preparation", "text":"Creating STL geometry and generating or importing routes are intentionally outside the generalized browser workflow. A case must already contain compatible geometry and authoritative routes. This prevents a site-specific LAZ, GIS or route-design procedure from being presented as a universal step."},
            {"heading":"No hidden calibration", "text":"Material defaults and forcing choices are documented. Validation observations are not used to tune the core physical formulation. Observation-informed atmospheric forcing, when selected explicitly, is recorded as forcing provenance rather than treated as geometry truth."},
            {"heading":"Naming", "note":"TREC-Route is short for Thermal Risk and Exposure Computation for Routes. Earlier output names may retain route_utci for file compatibility."},
        ]),
        diagram_page("Chapter 1 · purpose", "End-to-end computational workflow", workflow_diagram),
        text_page("Chapter 2 · cases and routes", "Self-contained case model", [
            {"heading":"Input/output isolation", "text":"Each problem is selected from input/<case> and writes only below run_output/<case>. case.json records file paths, location, timezone, OSM bounds, projected CRS, local origin and simulation defaults. case_config.py validates required files and prevents manifest paths from escaping the case directory."},
            {"heading":"Required prepared inputs", "bullets":["Building, vegetation and ground STL meshes.", "A routes directory containing route_<id>.csv/json, routes_index.json and route_polylines.pkl.", "Weather CSV; complete cached OSM features; OSM material and radiant-flux configuration.", "Optional radiation-forcing configuration for independent components or documented cloud treatment."]},
            {"heading":"General UI", "text":"The browser UI selects input and output cases, launches numbered stages 2–6, streams logs and tracks progress. Optional stage 4 solves diagnostic three-dimensional air temperature and velocity; skipping it preserves uniform shared-weather forcing. The numerical programs do not contain Lisbon- or MMC-specific case names."},
        ]),
        text_page("Chapter 2 · cases and routes", "Route geometry and coordinate contract", [
            {"heading":"Authoritative representation", "text":"route_<id>.csv is the authoritative ordered geometry. Required fields include sequence number, origin-shifted local X/Y, cumulative distance, projected coordinates and WGS84 latitude/longitude. JSON metadata records the route ID, name, length, number of points, CRS, local origin, source and generation arguments."},
            {"heading":"Coordinate chain", "formula":r"(x_{local},y_{local})+(x_0,y_0)=(x_{projected},y_{projected})\;\longleftrightarrow\;(lon,lat)"},
            {"heading":"Routing constraint", "text":"Planning routes follow the existing OSM pedestrian graph; free-space straight lines are rejected. Experimental mobile trajectories are an explicit exception because their measured coordinates must be preserved for exact observation pairing."},
            {"heading":"Invariance", "note":"OSM material polygons never change route graph nodes, edges, lengths, traversal order or walking time."},
        ]),
        diagram_page("Chapter 2 · cases and routes", "Route time is part of the exposure", route_timeline),
        text_page("Chapter 3 · ground materials", "Two independent uses of OpenStreetMap", [
            {"heading":"Pedestrian graph", "text":"The graph is authoritative only for route connectivity, candidate generation, length and traversal. It can be a compact walk-network representation."},
            {"heading":"Physical feature extract", "text":"A separate complete OSM cache retains nodes, ways, multipolygon relations and physical tags such as highway, surface, width, landuse, natural, leisure, amenity, parking and water. It exists to classify terrain materials, not to decide whether a route is walkable."},
            {"heading":"Caching and offline use", "text":"The exact domain plus configurable buffer is queried and cached. Existing caches are reused. A user-supplied GeoPackage, GeoJSON, OSM XML or PBF can replace Internet download. The workflow does not silently fall back to an incomplete graph-only material source."},
        ]),
        text_page("Chapter 3 · ground materials", "Classification, width and terrain partition", [
            {"heading":"Physical classes", "text":"Roads, pedestrian paths, sidewalks, crossings, plazas, parking, managed grass, sports/playground surfaces, bare ground, water and generic ground are recognized. surface=* is interpreted contextually; missing material tags use configurable class defaults and retain an uncertainty flag."},
            {"heading":"Finite width", "text":"Mapped polygons take precedence. Lines use width, estimated width, sidewalk width, optional lane inference and finally a class fallback. Buffering is rejected in geographic coordinates and always occurs in the scene’s projected metric CRS."},
            {"heading":"Exclusive assignment", "text":"Specific surfaces overwrite broad land cover by deterministic priority. Polygons are clipped and resolved before terrain assignment. Boundary-aware triangle overlap preserves narrow paths that centroid-only assignment would lose. Every terrain triangle receives exactly one persistent material ID."},
        ]),
    ]
    if materials:
        pages.append(material_table_page(materials))
    pages += [
        text_page("Chapter 4 · environmental forcing", "Weather and temporal interpolation", [
            {"heading":"Shared provider", "text":"Air temperature, relative humidity and wind come from one WeatherProvider used by MRT, UTCI and JOS-3. A CSV is preferred; parameterized fallback exists but its provenance is reported. Unit guards reject Kelvin-like air temperature, RH fractions supplied as percent, and invalid wind."},
            {"heading":"Periodic day", "text":"Stage 05 writes a timezone-aware times.csv containing solar position, DNI, DHI, GHI, Ta, RH, wind, cloud fraction, optional LWin and source metadata. Twenty-four-hour interpolation uses a periodic convention so arrival times passing midnight remain valid."},
            {"heading":"Wind interpretation", "text":"The surface solver consumes time-varying wind for convection. UTCI formally expects wind at 10 m; pedestrian-level wind must be converted consistently before it enters the workflow. JOS-3 receives the route-time wind directly as its local boundary condition."},
        ]),
        text_page("Chapter 4 · environmental forcing", "Solar and atmospheric-radiation forcing", [
            {"heading":"Solar position and clear sky", "text":"pvlib calculates apparent solar elevation and azimuth from date, latitude, longitude and timezone. The default shortwave boundary uses the Ineichen clear-sky model with an optional cloud adjustment. Independent measured/reference DNI, DHI and GHI can be supplied explicitly."},
            {"heading":"Mobile upper-envelope option", "text":"Mobile SWin is never replayed pointwise because it contains local building/tree shade and reflection already represented by ray tracing. An optional case mode compares only high open-sky-like SWin/clear-GHI ratios and infers one bounded session cloud attenuation. Insufficient samples fall back with a recorded warning."},
            {"heading":"Atmospheric longwave", "formula":r"L_{sky}=\epsilon_{sky}\,\sigma T_a^4"},
            {"text":"Clear-sky emissivity uses the humidity-dependent Prata formulation by default; cloud fraction blends it toward an overcast emissivity of 0.98. Independently supplied LWin overrides the parameterization where available."},
        ]),
        diagram_page("Chapter 5 · geometry and ray tracing", "Radiative visibility around a route receptor", ray_diagram),
        text_page("Chapter 5 · geometry and ray tracing", "Receptors, terrain height and acceleration", [
            {"heading":"Route sampling", "text":"Route polylines are sampled at configurable spacing. A downward ray finds local terrain height, and the receptor is placed at ground elevation plus the configured pedestrian height (default 1.1 m). This preserves sloped terrain rather than assuming z=0."},
            {"heading":"Intersection logic", "text":"Trimesh/Embree intersectors test full building, ground and vegetation meshes. Solid geometry blocks radiation. Vegetation intersections remain distinct because canopy transmits rather than behaving as an opaque building."},
            {"heading":"Performance", "text":"Static geometry arrays and per-timestep fields are stored as compact NumPy matrices. Batches bound memory. Expensive sky-view factors are cached only when mesh fingerprints, route points and sky-sampling parameters exactly match; --force-svf bypasses the cache."},
        ]),
        text_page("Chapter 5 · geometry and ray tracing", "Sky view and vegetation transmission", [
            {"heading":"Sky-view sampling", "text":"The upper hemisphere is sampled in azimuth/elevation. Planar weights describe a horizontal ground receiver; standing-cylinder weights describe a person and emphasize different directions. Stage 05a uses a full sphere to partition pedestrian view among sky, vegetation and first-hit surfaces."},
            {"heading":"Beer–Lambert canopy", "formula":r"\tau_{veg}=\exp(-k_{LAD}\,L)"},
            {"text":"The effective leaf-area path L is derived from vegetation intersections. Separate extinction coefficients are available for direct and diffuse radiation. A blocked solid ray contributes zero; an unobstructed sky ray contributes one; canopy produces an intermediate transmission."},
            {"heading":"Conservation", "text":"Directional weights are normalized. Stage 05a checks that sky, vegetation, surface and any documented default fractions sum to unity at every traced point."},
        ]),
        text_page("Chapter 6 · shortwave radiation", "Direct and diffuse absorbed shortwave", [
            {"heading":"Direct beam", "formula":r"K_{dir,abs}=\alpha_{sw}\,f_p(h)\,\tau_{dir}\,DNI"},
            {"text":"The default human shortwave absorptivity is 0.70. Direct visibility and canopy transmission are receptor-specific. The standing-person projected-area factor varies with solar elevation; a sphere option uses a constant factor."},
            {"heading":"Standing projected area", "formula":r"f_p(h)=0.308\cos\!\left[h\,(0.998-h^2/50000)\right]"},
            {"text":"The argument is evaluated in degrees as implemented from the Fanger/SOLWEIG-style expression. High sun exposes less projected area of a standing body than a sphere."},
            {"heading":"Diffuse sky", "formula":r"K_{dif,abs}=\alpha_{sw}\,f_{sky,dif}\,SVF_{person}\,DHI"},
        ]),
        text_page("Chapter 6 · shortwave radiation", "Local reflected shortwave", [
            {"heading":"Local illumination", "formula":r"K_{global,local}=\tau_{dir}DNI\sin h+SVF_{ground}DHI"},
            {"heading":"Absorbed reflection", "formula":r"K_{refl,abs}=\alpha_{sw}\,f_{ground}\,a_{local}\,K_{global,local}"},
            {"text":"The reflection term uses sunlight that actually reaches the nearby ground, not one domain-wide GHI value. Deep shade therefore receives little reflected shortwave. The local effective albedo is assembled from the material fractions of route-visible ground facets."},
            {"heading":"Current boundary", "note":"The current reflected-shortwave model attributes local ground reflection. It does not invent wall reflection when the radiation solver has not calculated it."},
        ]),
        text_page("Chapter 7 · route-visible thermal facets", "Why only route-relevant surfaces are solved", [
            {"heading":"Selection philosophy", "text":"Longwave impact depends on surfaces visible from, and sufficiently close to, the route. Stage 05a traces full-sphere rays from a stride-sampled subset of receptors. First-hit building/ground facets within the maximum distance form the thermal set and a sparse point-to-facet view matrix."},
            {"heading":"Full geometry still casts shade", "text":"Culling limits only the facets whose temperatures are solved. Direct-sun rays from those facets are always tested against complete building, ground and vegetation meshes, so a discarded thermal facet can still cast a correct shadow."},
            {"heading":"Interpolation", "text":"The route-visible longwave field is sampled at a coarser receptor stride because it is smoother than direct shadow transitions. Point maps and weights transfer the solved facet field back to all MRT receptors with explicit closure checks."},
        ]),
        diagram_page("Chapter 8 · surface energy", "Facet surface-energy balance", energy_diagram),
        text_page("Chapter 8 · surface energy", "Governing surface balance", [
            {"heading":"Surface equation", "formula":r"(1-a)K_{in}+\epsilon L_{in}-\epsilon\sigma T_s^4-h_c(T_s-T_a)-LE=q_{cond}"},
            {"text":"Each facet has its own incident solar radiation, sky fraction, material and evolving temperature. Direct solar uses full-geometry shade; diffuse and surrounding terms use the facet sky fraction and enclosure state."},
            {"heading":"Convection", "formula":r"h_c=5.7+3.8U\quad\mathrm{W\,m^{-2}K^{-1}}"},
            {"text":"The default McAdams relation uses the time-varying wind in times.csv. A Watmuff alternative and the previous film-subtracted form remain explicit comparison options."},
            {"heading":"Latent heat", "text":"Moisture-capable materials use a configurable Priestley–Taylor equilibrium approximation constrained by material evaporative efficiency and a positive-net-radiation cap. It is a first-order water-availability term, not a soil-moisture or plant-physiology model."},
        ]),
        text_page("Chapter 8 · surface energy", "One-dimensional substrate and time solution", [
            {"heading":"Thermal mass", "text":"Each material is represented by graded layers from the exposed surface to an effective depth. Conductance uses k and layer thickness; heat storage uses volumetric heat capacity C. Ground-like materials use a fixed deep-temperature boundary; walls and roofs use an insulated interior boundary."},
            {"heading":"Numerical method", "text":"Implicit Euler advances all facets of one material class together. Surface emission is Newton-linearized about the previous step. The tridiagonal layer system is solved with a vectorized Thomas algorithm, avoiding one dense solve per facet."},
            {"heading":"Diurnal spin-up", "text":"The forcing day repeats until the maximum facet cycle-end change falls below the configured tolerance or the maximum spin-up count is reached. The last complete cycle is saved, and spinup_report.txt records convergence instead of assuming it."},
            {"heading":"Energy diagnostics", "text":"Surface temperature, albedo, emissivity, direct transmission, latent heat, evaporative efficiency and material summaries are exported. Grey-radiosity closure is checked numerically."},
        ]),
        text_page("Chapter 8 · surface energy", "Optional diagnostic microclimate field", [
            {"heading":"Backward-compatible choice", "text":"The default boundary broadcasts the shared WeatherProvider air temperature and wind speed at each time. Optional UI step 4 instead solves route-buffered, spatially and temporally varying air temperature plus local X, local Y and vertical velocity. Relative humidity remains weather-forced."},
            {"heading":"Surface-to-air coupling", "formula":r"Q_H=h_c(T_s-T_a)"},
            {"text":"Sensible heat from each route-visible facet is deposited in its first exterior fluid cell. Positive heat seeds a vertical plume velocity proportional to the square root of buoyant temperature excess. A configured-direction logarithmic background wind supplies the synoptic flow because the legacy weather contract contains speed but not direction."},
            {"heading":"Mass consistency", "formula":r"-\nabla^2\phi=-\nabla\!\cdot\mathbf{u}^{*},\qquad\mathbf{u}=\mathbf{u}^{*}-\nabla\phi"},
            {"text":"An obstacle-aware finite-volume Poisson projection uses identical fluid faces for divergence and pressure correction. Terrain and buildings are impermeable; conjugate gradients solve the sparse elliptic system."},
        ]),
        text_page("Chapter 8 · surface energy", "Eulerian air-temperature transport", [
            {"heading":"Advection–diffusion", "formula":r"\frac{\partial T}{\partial t}+\mathbf{u}\!\cdot\nabla T=\kappa_t\nabla^2T+\frac{Q_H}{\rho c_p V_{cell}}"},
            {"text":"First-order upwind advection and constant effective eddy diffusivity advance air temperature. Automatic substeps enforce an advective/diffusive CFL limit. A cell-count cap adapts horizontal resolution to protect memory."},
            {"heading":"Staggered correction", "text":"The solver reads the baseline facet-temperature cycle, produces the 4-D air field, reruns stage 05b with local facet Ta and vector-speed magnitude, and regenerates MRT using local receptor Ta. Additional coupling iterations are configurable."},
            {"heading":"Scientific boundary", "note":"This is a mass-consistent diagnostic model, not RANS or LES. It has no prognostic momentum/turbulence closure, canopy drag or humidity transport. Wind direction, outer-boundary treatment, grid convergence and velocity predictions require independent validation."},
        ]),
        diagram_page("Chapter 9 · longwave radiation", "Energy-conserving grey-surface radiosity", radiosity_diagram),
        text_page("Chapter 9 · longwave radiation", "Grey enclosure equations", [
            {"heading":"Facet irradiation", "formula":r"G_i=f_{sky,i}L_{sky}+(1-f_{sky,i})\bar{J}"},
            {"heading":"Facet radiosity", "formula":r"J_i=\epsilon_i\sigma T_i^4+(1-\epsilon_i)G_i"},
            {"text":"Radiosity includes emitted longwave plus reflected incident longwave, satisfying Kirchhoff consistency for an opaque grey surface. The enclosure mean is weighted by facet area and non-sky fraction and solved analytically including repeated reflection."},
            {"heading":"Route-local approximation", "text":"Only stage-05a route-visible facets participate. The closure is a mean-field enclosure rather than a claimed full facet-to-facet view-factor matrix. Sky-facing isolated facets therefore do not dominate the street-level environment."},
            {"heading":"Source identity", "text":"The first-hit view matrix retains material/object class. Surface longwave can be attributed to asphalt, concrete, paving, generic ground, grass, walls, roofs, canopy, water and other available classes."},
        ]),
        text_page("Chapter 9 · longwave radiation", "Pedestrian longwave exposure", [
            {"heading":"Sky and surface absorption", "formula":r"L_{abs}=\epsilon_p\,[f_{sky,p}L_{sky}+(1-f_{sky,p})L_{surface}]"},
            {"text":"The default pedestrian longwave emissivity is 0.97. With facet thermal data, Lsurface is assembled from actual route-visible facet radiosities and vegetation/unresolved enclosure terms. The full-sphere sky fraction counts ground below the pedestrian; the legacy upper-hemisphere option can be retained for comparison."},
            {"heading":"Material coupling", "text":"A ray hit uses the local facet temperature and emissivity generated by the surface solver. Thus changing asphalt to concrete can change absorbed solar energy, surface temperature, radiosity and pedestrian MRT without altering route connectivity."},
            {"heading":"Legacy baseline", "text":"WITH_BASELINE=1 computes the older uniform-surround temperature result separately. It is opt-in because the authoritative facet-thermal run supersedes its MRT field."},
        ]),
        text_page("Chapter 10 · absorbed flux and MRT", "From absorbed radiation to mean radiant temperature", [
            {"heading":"Absorbed load", "formula":r"S_{abs}=K_{dir}+K_{dif}+K_{refl}+L_{sky}+L_{surface}"},
            {"heading":"MRT transformation", "formula":r"T_{mrt,K}=\left(\frac{S_{abs}}{\epsilon_p\sigma}\right)^{1/4}"},
            {"text":"MRT is the uniform black enclosure temperature that would produce the same absorbed radiant load for the reference body. The fourth-root transformation is nonlinear. Radiation mechanisms are therefore recorded and plotted in W m⁻²; they are never treated as additive MRT contributions in degrees Celsius."},
            {"heading":"Conservation", "text":"At every receptor and timestep, shortwave components must sum to total absorbed shortwave; sky plus surface longwave must sum to total longwave; and the combined total must reconstruct authoritative MRT within configured tolerances."},
        ]),
        text_page("Chapter 10 · absorbed flux and MRT", "Contribution data and source classification", [
            {"heading":"Primary mechanisms", "bullets":["Direct shortwave", "Diffuse-sky shortwave", "Surface-reflected shortwave", "Atmospheric/sky longwave", "Surface longwave"]},
            {"heading":"Material-resolved sources", "text":"Where source IDs are available, reflected shortwave and surface longwave are subdivided by every active class. Small categories can be combined into Other for legibility after verifying that aggregation preserves total flux."},
            {"heading":"Efficient storage", "text":"Rays are aggregated into receptor-level arrays in the inner calculation. Detailed ray-by-ray storage is avoided unless explicitly requested. The NPZ archive and metadata define units, categories and closure tolerances."},
        ]),
        text_page("Chapter 10 · absorbed flux and MRT", "Ranked contribution figures", [
            {"heading":"Independent ranking", "text":"For visualization only, every contribution series is copied and independently sorted from highest to lowest. Rank 1 for direct shortwave need not be the same receptor or time as rank 1 for surface longwave. Lines are not stacked because independently ranked components do not share simultaneous positions."},
            {"heading":"Preserved physical order", "text":"The exported receptor table stays in route order. Walking-time integration, cumulative exposure, UTCI, JOS-3, maps and rankings use that original order. Only a plot copy is sorted."},
            {"heading":"Separate longwave classification", "text":"The main primary-mechanism plot keeps surface longwave combined. A separate figure shows total surface longwave and all nonzero surface classifications, each independently descending."},
        ]),
        text_page("Chapter 11 · UTCI", "Universal Thermal Climate Index", [
            {"heading":"Inputs", "formula":r"UTCI=f(T_a,\;T_{mrt},\;v_{10m},\;RH)"},
            {"text":"TREC-Route uses the pythermalcomfort implementation of the operational Bröde et al. UTCI polynomial. MRT is spatially and temporally resolved; weather comes from the shared provider. Input checks document extrapolation outside the operational wind band."},
            {"heading":"Route outputs", "text":"For each route, UTCI is evaluated at every point’s arrival time. The software reports mean and peak UTCI, mean and peak MRT and strong-heat-stress dose."},
            {"heading":"Dose", "formula":r"D_{32}=\int_{walk}\max(0,UTCI-32^\circ C)\,dt"},
            {"text":"The implementation uses trapezoidal point intervals and reports degree-minutes above 32 °C."},
        ]),
        text_page("Chapter 11 · UTCI", "What UTCI does and does not mean", [
            {"heading":"Equivalent environment", "text":"UTCI is an equivalent temperature designed for outdoor thermal stress assessment under a standardized reference person and activity. It is not an observed air temperature and does not retain physiological state from one route point to the next."},
            {"heading":"Role in TREC-Route", "text":"UTCI provides an interpretable instantaneous exposure index and route ranking. Spatial variation is usually dominated by MRT when Ta, RH and wind are uniform or slowly varying, but the workflow exports all drivers so this can be checked rather than assumed."},
            {"heading":"Wind caveat", "note":"Supplying pedestrian-level wind directly to a formula defined for 10 m wind changes interpretation. Wind-height harmonization belongs in case forcing preparation and must be documented."},
        ]),
        text_page("Chapter 12 · JOS-3", "Sequential human thermoregulation", [
            {"heading":"Why a dynamic model", "text":"A person walking through shade and sun carries thermal state forward. JOS-3 is a multi-segment, multi-node thermoregulation model exposed through pythermalcomfort with an explicit time-stepping interface. It is used because steady one-environment comfort models cannot represent route history."},
            {"heading":"State and boundary conditions", "text":"A fresh JOS3 object is created for each route. Height, weight, age, sex, fat percentage and cardiac index define the subject. Air temperature, MRT, RH, wind and physical activity ratio drive each time step."},
            {"heading":"Equilibration", "text":"The model is first held at route-start conditions for a configurable period (default 10 minutes). The reported route rise starts after equilibration, preventing the library’s generic initial state from being mistaken for route-induced strain."},
        ]),
        text_page("Chapter 12 · JOS-3", "Core-temperature metric and subject profiles", [
            {"heading":"Whole-body summary", "formula":r"T_{core,WB}=\sum_s w_{BSA,s}\,T_{core,s},\qquad \sum_s w_{BSA,s}=1"},
            {"text":"JOS-3 returns segment core temperatures. TREC-Route forms a body-surface-area-weighted scalar at equilibration and at route end. Final core-temperature rise is their difference; segment traces remain available for physiological interpretation."},
            {"heading":"Route stepping", "text":"The same JOS-3 state advances using actual point-to-point travel durations. Resolved MRT and weather are sampled at each arrival time. Non-positive durations and invalid RH, wind or temperatures are rejected."},
            {"heading":"Profiles and uncertainty", "text":"Healthy adult, female, child, elderly, obese and acclimatized presets provide documented anthropometric/perfusion assumptions. A profile is not a complete clinical risk model; illness, medication, hydration and individual acclimatization remain outside the prediction."},
        ]),
        text_page("Chapter 13 · route analysis", "Timing, exposure and ranking", [
            {"heading":"Arrival schedule", "text":"Ordinary routes use cumulative distance divided by walking speed. Experimental trajectories may supply recorded elapsed time and local arrival hour, which take precedence. Route geometry and point order remain unchanged."},
            {"heading":"Environmental lookup", "text":"A cKDTree maps route coordinates to the nearest precomputed MRT receptor. Time interpolation is periodic across 24 hours. The same receptor/time indices sample absorbed-flux contribution matrices."},
            {"heading":"Ranking", "text":"The UTCI stage ranks from lower to higher mean UTCI, with peak UTCI as a tie-break. JOS-3 produces final core-temperature rise for physiological comparison. These are related but not interchangeable objectives."},
        ]),
        text_page("Chapter 13 · route analysis", "Uniform-reference sensitivity analysis", [
            {"heading":"Purpose", "text":"The standalone sensitivity program asks whether spatially varying air temperature and vapor pressure can be replaced by plausible uniform reference conditions while preserving resolved MRT, wind, timing and route geometry."},
            {"heading":"Humidity conversion", "formula":r"RH=100\,e/e_s(T_a),\quad e_s=6.112\exp\!\left(\frac{17.67T_a}{T_a+243.5}\right)"},
            {"heading":"Two outcome families", "text":"Ranking fidelity is assessed with best-route preservation, complete ordering, Spearman/Kendall statistics and pairwise contrast retention. Absolute accuracy is assessed with signed error, MAE, RMSE and prediction envelopes. Finite-difference slopes are labelled sensitivities over the tested range, not universal physiology."},
        ]),
        text_page("Chapter 14 · outputs and UI", "Output data model", [
            {"heading":"Numerical fields", "text":"mrt_facet_out stores receptor coordinates, time forcing, sky-view arrays, direct transmission, MRT matrices and absorbed-flux archives. thermal_out stores selected facets, view weights, surface temperatures, material properties, latent heat and radiosity."},
            {"heading":"Route products", "text":"Route UTCI and JOS-3 directories contain ranking summaries, per-point traces, maps and comparison figures. Routes are exported as CSV, GeoJSON and GPX for GIS and external validation. Ground-material outputs include GeoPackage layers, face IDs, material catalogs and route-point downward-hit diagnostics."},
            {"heading":"Provenance", "text":"Weather, radiation forcing, material assumptions, cache identity, interpolation closure and validation tolerances are written beside results. Console result markers and output paths support UI progress and reproducible automation."},
        ]),
        text_page("Chapter 14 · outputs and UI", "Browser workflow and progress", [
            {"heading":"Step 1", "text":"The user selects an input case and an output case. Path overrides that could leak outputs into another case are blocked by the UI transport."},
            {"heading":"Executable steps", "bullets":["2 — OSM data and ground-material subdivision.", "3 — MRT preparation, visible facets, energy balance and authoritative facet-thermal MRT.", "4 — optional diagnostic 3-D Ta/velocity, surface recoupling and regenerated MRT.", "5 — MRT and UTCI visualizations.", "6 — route UTCI, JOS-3 and optional case-specific comparison."]},
            {"heading":"Run controls", "text":"Only-this-step mode fences later sub-stages with SKIP flags; this-and-after runs the selected stage through completion. Progress bars consume explicit workflow events and numerical batch counters rather than elapsed-time guesses."},
        ]),
        diagram_page("Chapter 15 · verification", "Verification and validation ladder", validation_diagram),
        text_page("Chapter 15 · verification", "Conservation and automated checks", [
            {"heading":"Analytic/numerical tests", "bullets":["Directional weights partition unity; an isothermal enclosure reproduces σT⁴.", "Grey radiosity satisfies J=εσT⁴+(1−ε)G and repeated-reflection closure.", "Batched tridiagonal solutions match dense solutions.", "Steady conduction and nonlinear surface equilibrium conserve energy.", "Absorbed-flux components reconstruct total flux and authoritative MRT."]},
            {"heading":"Integration tests", "text":"Synthetic scenes check sun/shade contrast, nighttime cooling, material-specific response, narrow path preservation, route-graph invariance, sorting-copy isolation and disabled-mode backward compatibility."},
            {"heading":"Fail loudly", "text":"Missing files, CRS errors, malformed routes, invalid RH/wind/temperature, shape mismatches, non-converged conservation identities and changed protected route artifacts generate explicit errors or warnings."},
        ]),
        text_page("Chapter 16 · validation", "Independent comparisons", [
            {"heading":"Validation cases", "text":"The project includes fixed-point and mobile datasets associated with SOLWEIG Gothenburg sites, a Bauhaus-University Weimar ENVI-met dataset and Lisbon mobile measurements. Case-specific comparison scripts remain outside the general UI workflow."},
            {"heading":"Statistics", "text":"Where supported, comparisons report paired count, mean bias, MAE, RMSE, normalized RMSE, R², Pearson correlation, Willmott agreement, regression slope/intercept and subgroups by point, day, sun/shade or period."},
            {"heading":"Flux before temperature", "text":"Four-component radiometer data can be converted into an explicitly labelled two-hemisphere absorbed-flux diagnostic. It is not identical to a six-directional standing-person measurement, but it helps distinguish shortwave and longwave bias without inventing additive MRT terms."},
        ]),
        text_page("Chapter 16 · validation", "Uncertainty and known limitations", [
            {"heading":"Geometry", "text":"STL quality, tree representation, measurement height, north orientation and narrow-object resolution control sun/shade transitions. Surface material improvements cannot compensate for missing physical obstruction geometry."},
            {"heading":"OSM and materials", "text":"OSM tags and widths may be incomplete or wrong. Default albedo, emissivity and thermal properties are assumptions until verified by orthophoto, field survey or site metadata. The latent model does not predict soil moisture."},
            {"heading":"Forcing", "text":"Clear-sky decomposition, cloud inference, atmospheric longwave, wind height and temporal averaging can dominate error. Mobile solar observations include local shade and must not be treated as domain-wide point forcing."},
            {"heading":"Human models", "text":"UTCI represents a standardized equivalent environment. JOS-3 predictions depend on subject, activity, clothing/library defaults, hydration and health assumptions. Outputs support risk comparison but are not medical diagnosis."},
        ]),
        text_page("Chapter 16 · validation", "How to interpret disagreement", [
            {"heading":"Mechanism sequence", "text":"First verify time/date/timezone and units. Then compare atmospheric GHI/LWin, sun/shade state, absorbed shortwave, absorbed longwave, local material and surface temperature. Only after these should total MRT or physiological error be interpreted."},
            {"heading":"Sensor equivalence", "text":"A globe thermometer, net radiometer and numerical standing body do not share identical directional response or time constant. Conversion equations, globe size/emissivity/convection and sensor averaging must be documented rather than tuned to agreement."},
            {"heading":"Route versus absolute result", "note":"A route ranking can be robust even when all routes share appreciable absolute bias. Conversely, a small pooled RMSE can conceal locally reversed route contrasts."},
        ]),
        text_page("Chapter 17 · reproducibility", "Running and reproducing a case", [
            {"heading":"Typical command", "formula":r"\mathtt{INPUT\_CASE\_DIR=input/CASE\;OUTPUT\_CASE\_DIR=run\_output/CASE\;bash\;start.sh\;2}"},
            {"heading":"Controlled overrides", "text":"Date, departure hour, walking speed, subject profile, weather, local origin, projected CRS, cloud, sampling resolution, facet distance and spin-up settings can be supplied through documented environment variables or stage arguments. Defaults remain recorded in case.json and output metadata."},
            {"heading":"Caches", "text":"Complete OSM downloads and SVF fields are cached. Cache sidecars record bounds, retrieval date or geometry fingerprints and sky-sampling inputs. Exact identity is required before reuse."},
            {"heading":"Recommended archive", "text":"Preserve the input case, case.json, configuration JSON, raw OSM cache metadata, program version, console log and complete output case. Do not archive only final figures."},
        ]),
        text_page("Appendix A · symbols", "Principal symbols and units", [
            {"heading":"Radiation", "bullets":["DNI, DHI, GHI — direct normal, diffuse horizontal and global horizontal shortwave irradiance [W m⁻²].", "K — shortwave radiation; L — longwave radiation; Sabs — total absorbed radiant flux [W m⁻²].", "a — shortwave albedo; αsw — human shortwave absorptivity; ε — longwave emissivity; σ — Stefan–Boltzmann constant.", "SVF — sky-view factor; τ — direct/canopy transmission; fp — projected-area factor."]},
            {"heading":"Thermal state", "bullets":["Ta, Ts, Tmrt — air, surface and mean radiant temperature [°C or K as indicated].", "h — convective coefficient [W m⁻² K⁻¹]; k — conductivity [W m⁻¹ K⁻¹]; C — volumetric heat capacity [J m⁻³ K⁻¹].", "LE — latent heat flux [W m⁻²]; qcond — conductive flux into the substrate [W m⁻²]."]},
            {"heading":"Route and physiology", "bullets":["UTCI — Universal Thermal Climate Index [°C].", "PAR — physical activity ratio; BSA — body surface area; Tcore — JOS-3 core temperature [°C].", "Distance [m], time [s or h], wind [m s⁻¹], RH [%]."]},
        ]),
        text_page("Appendix B · file map", "Principal implementation modules", [
            {"heading":"Workflow", "text":"start.sh; pipeline_ui.py; case_config.py; weather_provider.py; physical_checks.py."},
            {"heading":"Geometry and physical surfaces", "text":"download_osm_complete_features.py; osm_ground_materials.py; prepare_osm_ground_materials.py; route_ground_material_diagnostics.py; external case-specific geometry utilities."},
            {"heading":"Radiation and surfaces", "text":"05_mrt_network_raytrace.py; 05a_thermal_facets_select.py; 05b_facet_energy_balance.py; thermal_common.py; radiant_flux_contributions.py; radiation_forcing.py."},
            {"heading":"Routes and human outcomes", "text":"generate_route.py; 08_route_thermal_stress.py; 09_route_thermal_stress_jos3.py; subject_profiles.py; sensitivity.py."},
            {"heading":"Verification", "text":"verify_thermal_pipeline.py; verify_osm_ground_materials.py; verify_radiant_flux_contributions.py; verify_pipeline_ui.py and independent case comparison scripts."},
        ]),
        text_page("References", "Physical and computational references I", [
            {"heading":"Thermal radiation and outdoor comfort", "bullets":["Fanger, P. O. (1972). Thermal Comfort. McGraw-Hill.", "ISO 7726 (1998). Ergonomics of the thermal environment — Instruments for measuring physical quantities.", "VDI 3787 Part 2. Environmental meteorology: methods for the human-biometeorological evaluation of climate and air quality for urban and regional planning.", "Lindberg, F., Holmer, B. & Thorsson, S. (2008). SOLWEIG 1.0 — modelling spatial variations of 3-D radiant fluxes and mean radiant temperature in complex urban settings. International Journal of Biometeorology, 52, 697–713.", "Prata, A. J. (1996). A new long-wave formula for estimating downward clear-sky radiation at the surface. Quarterly Journal of the Royal Meteorological Society, 122, 1127–1151."]},
            {"heading":"Solar and surface energy", "bullets":["Ineichen, P. & Perez, R. (2002). A new airmass independent formulation for the Linke turbidity coefficient. Solar Energy, 73(3), 151–157.", "Priestley, C. H. B. & Taylor, R. J. (1972). On the assessment of surface heat flux and evaporation using large-scale parameters. Monthly Weather Review, 100, 81–92.", "McAdams, W. H. (1954). Heat Transmission, 3rd edition. McGraw-Hill."]},
        ]),
        text_page("References", "Physical and computational references II", [
            {"heading":"Thermal indices and physiology", "bullets":["Bröde, P. et al. (2012). Deriving the operational procedure for the Universal Thermal Climate Index (UTCI). International Journal of Biometeorology, 56, 481–494.", "Kobayashi, Y. & Tanabe, S. (2013). Development of JOS-2 human thermoregulation model with detailed vascular system. Building and Environment, 66, 1–10.", "Tartarini, F. & Schiavon, S. (2020). pythermalcomfort: A Python package for thermal comfort research. SoftwareX, 12, 100578."]},
            {"heading":"Software and geospatial foundations", "bullets":["Holmgren, W. F. et al. (2018). pvlib python: a Python package for modeling solar energy systems. Journal of Open Source Software, 3(29), 884.", "Boeing, G. (2017). OSMnx: new methods for acquiring, constructing, analyzing, and visualizing complex street networks. Computers, Environment and Urban Systems, 65, 126–139.", "OpenStreetMap contributors. OpenStreetMap vector data and tagging schema. Individual cached extracts retain retrieval metadata."]},
            {"heading":"Implementation note", "note":"References identify the scientific lineage and supporting software. The equations and defaults documented in this book correspond to the current local TREC-Route implementation and its configuration files; they should be cited alongside the relevant primary method sources."},
        ]),
    ]
    if selected_case is not None:
        pages.append(case_page(selected_case.resolve()))
    return pages


def chapter_entries(pages: list[Page], first_content_page: int) -> list[tuple[str, int]]:
    entries=[]; seen=set()
    for offset,page in enumerate(pages):
        if page.chapter not in seen:
            entries.append((page.chapter, first_content_page + offset))
            seen.add(page.chapter)
    return entries


def generate(args: argparse.Namespace) -> Path:
    output=args.output.resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    materials=load_materials()
    pages=build_pages(materials,args.case)
    toc_parts=max(1,int(np.ceil(len(chapter_entries(pages,1))/12)))
    first_content_page=2+toc_parts
    entries=chapter_entries(pages,first_content_page)
    total_pages=1+toc_parts+len(pages)
    generated=datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    ctx=BookContext(args.author,args.version,generated,total_pages)
    metadata={"Title":f"{args.title}: Implementation and Physics",
              "Author":args.author,
              "Subject":"TREC-Route software methods, physics and implementation",
              "Keywords":"outdoor thermal comfort, MRT, UTCI, JOS-3, radiation, routes"}
    with PdfPages(output,metadata=metadata) as pdf:
        ctx.page_number=1
        fig=plt.figure(figsize=A4); cover(fig,ctx,args.title); pdf.savefig(fig); plt.close(fig)
        for part in range(1,toc_parts+1):
            ctx.page_number=1+part
            fig=plt.figure(figsize=A4); contents(fig,ctx,entries,part,toc_parts); pdf.savefig(fig); plt.close(fig)
        for index,page in enumerate(pages,first_content_page):
            ctx.page_number=index
            fig=plt.figure(figsize=A4); page_frame(fig,page,ctx); page.renderer(fig,ctx)
            pdf.savefig(fig); plt.close(fig)
    print(f"Generated {output}")
    print(f"Pages: {total_pages}")
    print("Contents:")
    for label,page_number in entries:
        print(f"  {page_number:>3}  {label}")
    return output


def main() -> int:
    args=parse_args()
    if args.case is not None:
        candidate=args.case / "case.json" if args.case.is_dir() else args.case
        if not candidate.is_file():
            raise FileNotFoundError(f"selected case manifest not found: {candidate}")
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
