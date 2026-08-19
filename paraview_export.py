"""paraview_export.py -- dependency-free VTK XML writers for ParaView.

TREC-Route's optional step 4 stores its results as raw ``.npy``/``.npz``
arrays, which ParaView cannot open directly.  This module writes the two
VTK XML container types those results need:

  * ``write_vtr``  -- RectilinearGrid (``.vtr``) for the structured 3-D
    air-temperature/velocity field of stage 05c.  Field values live at the
    cell-center coordinates stored in ``microclimate_axes.npz`` and are
    written as PointData at exactly those coordinates, matching the
    trilinear interpolation contract of ``microclimate_field.py``.
  * ``write_vtp``  -- PolyData (``.vtp``) for triangle meshes with
    per-facet (CellData) values: the stage-05d urban surface results and
    static context geometry.
  * ``write_pvd``  -- a ParaView collection that binds the per-timestep
    files into one animatable time series.

Only numpy and the standard library are used.  Appended data blocks are
zlib-compressed by default (``compressor="vtkZLibDataCompressor"``), which
matters at the real problem scale (millions of facets per timestep).

``read_vtk_appended`` reads back files produced by these writers so the
verification suites can round-trip byte-exact arrays; it is not a general
VTK reader.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ElementTree
import zlib

import numpy as np


_VTK_TYPES = {
    np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64",
    np.dtype(np.int32): "Int32", np.dtype(np.int64): "Int64",
    np.dtype(np.uint8): "UInt8", np.dtype(np.int16): "Int16",
}
_NUMPY_TYPES = {name: dtype for dtype, name in _VTK_TYPES.items()}


def _as_exportable(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    if array.dtype not in _VTK_TYPES:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


class _AppendedBlocks:
    """Accumulates appended-format data blocks and their byte offsets."""

    def __init__(self, compress: bool):
        self.compress = bool(compress)
        self.chunks: list[bytes] = []
        self.size = 0

    def add(self, array: np.ndarray) -> int:
        raw = array.tobytes()
        if self.compress:
            compressed = zlib.compress(raw, 1)
            # Single-block vtkZLibDataCompressor header (UInt64 header_type):
            # [n_blocks, block_size, last_block_size, compressed_size].
            header = np.array([1, len(raw), len(raw), len(compressed)],
                              dtype="<u8").tobytes()
            chunk = header + compressed
        else:
            chunk = np.array([len(raw)], dtype="<u8").tobytes() + raw
        offset = self.size
        self.chunks.append(chunk)
        self.size += len(chunk)
        return offset


def _data_array(name: str | None, array: np.ndarray, components: int,
                blocks: _AppendedBlocks) -> str:
    offset = blocks.add(array)
    label = f' Name="{name}"' if name else ""
    return (f'<DataArray type="{_VTK_TYPES[array.dtype]}"{label} '
            f'NumberOfComponents="{components}" format="appended" '
            f'offset="{offset}"/>')


def _write_file(path: Path, grid_xml: str, dataset_type: str,
                blocks: _AppendedBlocks) -> None:
    compressor = (' compressor="vtkZLibDataCompressor"'
                  if blocks.compress else "")
    header = (f'<?xml version="1.0"?>\n'
              f'<VTKFile type="{dataset_type}" version="1.0" '
              f'byte_order="LittleEndian" header_type="UInt64"{compressor}>\n'
              f'{grid_xml}\n<AppendedData encoding="raw">\n_')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for chunk in blocks.chunks:
            stream.write(chunk)
        stream.write(b"\n</AppendedData>\n</VTKFile>\n")


def _named_arrays(data: dict | None, n_expected: int, kind: str,
                  blocks: _AppendedBlocks) -> str:
    lines = []
    for name, values in (data or {}).items():
        values = _as_exportable(values)
        # A scalar field may arrive in any shape (e.g. (z, y, x)); a vector
        # field carries its components on the LAST axis (e.g. (z, y, x, 3)).
        if values.size == n_expected:
            values = values.reshape(n_expected, 1)
        elif (values.ndim >= 2 and values.size
                == n_expected * values.shape[-1]):
            values = values.reshape(n_expected, values.shape[-1])
        else:
            raise ValueError(
                f"{kind} array {name!r} has {values.size} values, "
                f"expected a multiple of {n_expected} tuples")
        lines.append(_data_array(name, values, values.shape[1], blocks))
    return "\n".join(lines)


def write_vtr(path: str | Path, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              point_data: dict | None = None, compress: bool = True) -> None:
    """Write a RectilinearGrid with PointData sampled at (x, y, z).

    ``point_data`` values may be shaped ``(z, y, x)`` or ``(z, y, x, ncomp)``
    (the stage-05c array order); C-order flattening yields the x-fastest
    ordering VTK requires.
    """
    x, y, z = (np.asarray(axis, dtype=np.float32) for axis in (x, y, z))
    blocks = _AppendedBlocks(compress)
    extent = f"0 {len(x) - 1} 0 {len(y) - 1} 0 {len(z) - 1}"
    coordinates = "\n".join(
        _data_array(name, axis, 1, blocks)
        for name, axis in (("x_m", x), ("y_m", y), ("z_m", z)))
    points_xml = _named_arrays(point_data, len(x) * len(y) * len(z),
                               "PointData", blocks)
    grid = (f'<RectilinearGrid WholeExtent="{extent}">\n'
            f'<Piece Extent="{extent}">\n'
            f'<Coordinates>\n{coordinates}\n</Coordinates>\n'
            f'<PointData>\n{points_xml}\n</PointData>\n'
            f'</Piece>\n</RectilinearGrid>')
    _write_file(Path(path), grid, "RectilinearGrid", blocks)


def write_vtp(path: str | Path, vertices: np.ndarray, faces: np.ndarray,
              cell_data: dict | None = None, point_data: dict | None = None,
              compress: bool = True) -> None:
    """Write a triangle mesh with per-face CellData as VTK PolyData."""
    vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32))
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be (n_points, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (n_faces, 3) triangles")
    if len(faces) and int(faces.max()) >= len(vertices):
        raise ValueError("face indices exceed vertex count")
    connectivity = np.ascontiguousarray(faces, dtype=np.int32).reshape(-1)
    offsets = np.arange(3, 3 * len(faces) + 1, 3, dtype=np.int32)
    blocks = _AppendedBlocks(compress)
    points = _data_array(None, vertices, 3, blocks)
    polys = (_data_array("connectivity", connectivity, 1, blocks) + "\n"
             + _data_array("offsets", offsets, 1, blocks))
    cells_xml = _named_arrays(cell_data, len(faces), "CellData", blocks)
    points_xml = _named_arrays(point_data, len(vertices), "PointData", blocks)
    grid = (f'<PolyData>\n'
            f'<Piece NumberOfPoints="{len(vertices)}" NumberOfVerts="0" '
            f'NumberOfLines="0" NumberOfStrips="0" '
            f'NumberOfPolys="{len(faces)}">\n'
            f'<Points>\n{points}\n</Points>\n'
            f'<Polys>\n{polys}\n</Polys>\n'
            f'<CellData>\n{cells_xml}\n</CellData>\n'
            f'<PointData>\n{points_xml}\n</PointData>\n'
            f'</Piece>\n</PolyData>')
    _write_file(Path(path), grid, "PolyData", blocks)


def write_pvd(path: str | Path, entries: list[tuple[float, str]]) -> None:
    """Write a ParaView collection: [(time_value, relative_file), ...]."""
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" '
             'byte_order="LittleEndian">',
             "<Collection>"]
    for time_value, relative in entries:
        lines.append(f'<DataSet timestep="{float(time_value):.6f}" group="" '
                     f'part="0" file="{relative}"/>')
    lines += ["</Collection>", "</VTKFile>", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def read_vtk_appended(path: str | Path) -> dict:
    """Read one file written by this module back into named numpy arrays.

    Returns ``{"header": parsed XML Element, "arrays": {name: array}}``
    where unnamed Points arrays appear as ``"__points__"``.  Verification
    helper only -- supports exactly the layout this module writes.
    """
    data = Path(path).read_bytes()
    marker = data.index(b"<AppendedData")
    document = ElementTree.fromstring(data[:marker] + b"</VTKFile>")
    blob_start = data.index(b"_", marker) + 1
    compressed = document.get("compressor") == "vtkZLibDataCompressor"
    arrays: dict[str, np.ndarray] = {}
    for element in document.iter("DataArray"):
        dtype = _NUMPY_TYPES[element.get("type")]
        components = int(element.get("NumberOfComponents", "1"))
        position = blob_start + int(element.get("offset"))
        if compressed:
            header = np.frombuffer(data, "<u8", 4, position)
            n_blocks, raw_size, _, compressed_size = (int(v) for v in header)
            if n_blocks != 1:
                raise ValueError("reader supports single-block arrays only")
            payload = zlib.decompress(
                data[position + 32:position + 32 + compressed_size])
            if len(payload) != raw_size:
                raise ValueError(f"decompressed size mismatch in {path}")
        else:
            raw_size = int(np.frombuffer(data, "<u8", 1, position)[0])
            payload = data[position + 8:position + 8 + raw_size]
        values = np.frombuffer(payload, dtype=dtype)
        if components > 1:
            values = values.reshape(-1, components)
        name = element.get("Name")
        if name is None:
            name = "__points__" if element.tag == "DataArray" else name
        arrays[name] = values
    return {"header": document, "arrays": arrays}
