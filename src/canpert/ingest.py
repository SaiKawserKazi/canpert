"""Ranged, resumable ingestion of single cells from remote dense Perturb-seq h5ad files.

The four source files (Replogle 2022 K562/RPE1 on Figshare, Nadig 2025 HepG2/Jurkat on
GEO) store ``X`` as an uncompressed, contiguous float32 matrix. The byte offset of cell
``i`` is therefore ``offset(X) + i * n_genes * 4``. We read only the obs/var metadata,
select a stratified subsample of cells (all perturbations, capped cells each, plus a
control pool), and fetch those rows with parallel HTTP range requests. Rows are stored as
sparse integer UMI counts, so the full 30+ GB of source files never touch the disk.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import os
import threading
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

SOURCES: dict[str, str] = {
    "k562": "https://ndownloader.figshare.com/files/35773219",
    "rpe1": "https://ndownloader.figshare.com/files/35775606",
    "hepg2": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/"
    "GSE264667_hepg2_raw_singlecell_01.h5ad",
    "jurkat": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/"
    "GSE264667_jurkat_raw_singlecell_01.h5ad",
}
CONTROL_LABEL = "non-targeting"


@dataclass(frozen=True)
class Layout:
    """Byte layout of a remote dense matrix."""

    offset: int
    n_cells: int
    n_genes: int


def _decode(arr: np.ndarray) -> np.ndarray:
    return np.array([a.decode() if isinstance(a, bytes) else str(a) for a in arr])


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests.

    Features: a disk-backed block cache (restarts never refetch a block), parallel
    read-ahead of the next blocks, and resumable retries. Every request goes to the
    original URL, so short-lived signed redirects (Figshare S3 links expire after 10 s)
    are re-resolved each time.
    """

    def __init__(self, url: str, cache_dir: Path, block: int = 1 << 18, ahead: int = 12) -> None:
        super().__init__()
        self.url, self.block, self.pos, self.ahead = url, block, 0, ahead
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.local = threading.local()
        size_file = cache_dir / "size"
        if size_file.exists():
            self.size = int(size_file.read_text())
        else:
            head = _fetch_head(url)
            self.size = head
            size_file.write_text(str(head))
        self.mem: dict[int, bytes] = {}
        self.pool = cf.ThreadPoolExecutor(ahead)
        self.pending: dict[int, cf.Future[bytes]] = {}

    def _session(self) -> requests.Session:
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        return self.local.s  # type: ignore[no-any-return]

    def _load(self, b: int) -> bytes:
        path = self.cache_dir / f"{b}.bin"
        if path.exists():
            return path.read_bytes()
        lo = b * self.block
        data = _fetch(self.url, lo, min(lo + self.block, self.size), self._session())
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return data

    def readable(self) -> bool:  # noqa: D102
        return True

    def seekable(self) -> bool:  # noqa: D102
        return True

    def tell(self) -> int:  # noqa: D102
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:  # noqa: D102
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, b: int) -> bytes:
        n_blocks = (self.size + self.block - 1) // self.block
        for k in range(b, min(b + self.ahead, n_blocks)):
            if k not in self.mem and k not in self.pending:
                self.pending[k] = self.pool.submit(self._load, k)
        if b not in self.mem:
            self.mem[b] = self.pending.pop(b).result()
        return self.mem[b]

    def readinto(self, buf: memoryview) -> int:  # type: ignore[override]  # noqa: D102
        n = min(len(buf), self.size - self.pos)
        got = 0
        while got < n:
            b, off = divmod(self.pos + got, self.block)
            chunk = self._block(b)[off:off + n - got]
            buf[got:got + len(chunk)] = chunk
            got += len(chunk)
        self.pos += n
        return n

    def close(self) -> None:  # noqa: D102
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().close()


def _fetch_head(url: str) -> int:
    for attempt in range(500):
        try:
            r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
            if r.status_code == 206:
                return int(r.headers["Content-Range"].split("/")[-1])
        except requests.RequestException:
            pass
        time.sleep(min(60, 2 ** min(attempt, 6)))
    raise RuntimeError("size probe failed")


def read_metadata(url: str, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Layout]:
    """Read obs, var and X layout from a remote h5ad without touching X."""
    with RangeFile(url, cache_dir) as f, h5py.File(f, "r") as h:
        x = h["X"]
        assert x.chunks is None and x.compression is None, "X must be contiguous"
        assert x.dtype == np.float32
        layout = Layout(int(x.id.get_offset()), int(x.shape[0]), int(x.shape[1]))

        def table(group: str, keep: set[str] | None) -> pd.DataFrame:
            g = h[group]
            cats = g["__categories"] if "__categories" in g else {}
            cols: dict[str, np.ndarray] = {}
            for key in g:
                if key == "__categories" or not isinstance(g[key], h5py.Dataset):
                    continue
                if keep is not None and key not in keep:
                    continue
                vals = g[key][:]
                if key in cats:
                    labels = _decode(cats[key][:])
                    vals = np.where(vals >= 0, labels[np.clip(vals, 0, None)], "")
                elif vals.dtype == object:
                    vals = _decode(vals)
                cols[key] = vals
            return pd.DataFrame(cols)

        obs = table("obs", {"gene", "gem_group", "UMI_count", "mitopercent"})
        var = table("var", {"gene_id", "gene_name", "mean", "std"})
    return obs, var, layout


def select_cells(
    obs: pd.DataFrame, perts: set[str], max_cells: int, n_ctrl: int, seed: int
) -> np.ndarray:
    """Stratified subsample: <= max_cells per kept perturbation + n_ctrl controls."""
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for gene, idx in obs.groupby("gene").indices.items():
        if gene == CONTROL_LABEL:
            k = n_ctrl
        elif gene in perts:
            k = max_cells
        else:
            continue
        idx = np.asarray(idx)
        keep.append(rng.choice(idx, size=min(k, idx.size), replace=False))
    return np.sort(np.concatenate(keep))


def _ranges(rows: np.ndarray, max_gap: int, max_rows: int) -> list[tuple[int, int]]:
    """Merge sorted row indices into [start, stop) runs allowing small gaps."""
    out: list[tuple[int, int]] = []
    start = prev = int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r - prev <= max_gap and r - start < max_rows:
            prev = r
            continue
        out.append((start, prev + 1))
        start = prev = r
    out.append((start, prev + 1))
    return out


_RESOLVED: dict[str, tuple[str, float]] = {}
_RESOLVE_LOCK = threading.Lock()


def _resolve(url: str) -> str:
    """Cache the CDN redirect target of Hugging Face URLs for 20 min (saves a round trip
    per request); other hosts are used as given."""
    if "huggingface.co" not in url:
        return url
    with _RESOLVE_LOCK:
        hit = _RESOLVED.get(url)
        if hit and time.time() - hit[1] < 1200:
            return hit[0]
        try:
            r = requests.head(url, allow_redirects=False, timeout=60)
            loc = r.headers.get("Location", url)
        except requests.RequestException:
            loc = url
        _RESOLVED[url] = (loc, time.time())
        return loc


def _fetch(url: str, lo: int, hi: int, session: requests.Session, max_attempts: int = 500) -> bytes:
    """Fetch bytes [lo, hi) with resume-on-drop; tolerates outages of several hours."""
    buf = bytearray()
    for attempt in range(max_attempts):
        start = lo + len(buf)
        target = _resolve(url)
        try:
            with session.get(target, headers={"Range": f"bytes={start}-{hi - 1}"}, timeout=60,
                             stream=True) as resp:
                if resp.status_code != 206:
                    _RESOLVED.pop(url, None)  # expired or rejected link: re-resolve
                if resp.status_code == 206:
                    for chunk in resp.iter_content(1 << 16):
                        buf += chunk
                        if len(buf) >= hi - lo:
                            break
        except (requests.RequestException, OSError):
            pass
        if len(buf) >= hi - lo:
            return bytes(buf[: hi - lo])
        time.sleep(min(60, 2 ** min(attempt, 6)))
    raise RuntimeError(f"range {lo}-{hi} failed after {max_attempts} attempts")


def fetch_rows(
    url: str, layout: Layout, rows: np.ndarray, out_dir: Path, threads: int = 16
) -> None:
    """Fetch selected rows into resumable sparse shards under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    row_bytes = layout.n_genes * 4
    runs = _ranges(rows, max_gap=2, max_rows=64)
    # group runs into shards of ~500 rows each for resumability
    shards: list[list[tuple[int, int]]] = [[]]
    count = 0
    for run in runs:
        shards[-1].append(run)
        count += run[1] - run[0]
        if count >= 500:
            shards.append([])
            count = 0
    shards = [s for s in shards if s]
    rowset = set(rows.tolist())
    local = threading.local()

    def session() -> requests.Session:
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s  # type: ignore[no-any-return]

    def do_run(run: tuple[int, int]) -> tuple[np.ndarray, sp.csr_matrix]:
        lo = layout.offset + run[0] * row_bytes
        hi = layout.offset + run[1] * row_bytes
        buf = np.frombuffer(_fetch(url, lo, hi, session()), dtype=np.float32)
        mat = buf.reshape(run[1] - run[0], layout.n_genes)
        idx = np.arange(run[0], run[1])
        mask = np.array([i in rowset for i in idx])
        dense = mat[mask]
        assert np.all(dense >= 0) and np.allclose(dense, np.round(dense)), "not UMI counts"
        return idx[mask], sp.csr_matrix(dense.astype(np.int32))

    t0, done_bytes = time.time(), 0
    with cf.ThreadPoolExecutor(threads) as pool:
        for si, shard in enumerate(shards):
            path = out_dir / f"shard_{si:05d}.npz"
            if path.exists():
                continue
            results = list(pool.map(do_run, shard))
            ids = np.concatenate([r[0] for r in results])
            mat = sp.vstack([r[1] for r in results]).tocsr()
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
            np.savez_compressed(
                tmp, rows=ids, data=mat.data, indices=mat.indices, indptr=mat.indptr,
                shape=np.array(mat.shape),
            )
            tmp.replace(path)
            done_bytes += sum(r[1] - r[0] for r in shard) * row_bytes
            rate = done_bytes / max(time.time() - t0, 1e-9) / 1e6
            print(f"{out_dir.name} shard {si + 1}/{len(shards)} {rate:.2f} MB/s", flush=True)


def load_shards(out_dir: Path) -> tuple[np.ndarray, sp.csr_matrix]:
    """Concatenate fetched shards into (row ids, csr counts)."""
    ids, mats = [], []
    for path in sorted(out_dir.glob("shard_*.npz")):
        if ".tmp" in path.name:
            continue
        z = np.load(path)
        ids.append(z["rows"])
        mats.append(sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])))
    return np.concatenate(ids), sp.vstack(mats).tocsr()


def save_metadata(name: str, root: Path) -> None:
    """Stage 1: fetch obs/var/layout for one line."""
    d = root / name
    obs, var, layout = read_metadata(SOURCES[name], d / "meta_blocks")
    d.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(d / "obs.parquet")
    var.to_parquet(d / "var.parquet")
    (d / "layout.json").write_text(json.dumps(layout.__dict__))


def shared_panel(root: Path, names: tuple[str, ...]) -> list[str]:
    """Perturbations present (with >= 1 cell) in every line's obs."""
    sets = []
    for n in names:
        obs = pd.read_parquet(root / n / "obs.parquet", columns=["gene"])
        sets.append(set(obs["gene"].astype(str)) - {CONTROL_LABEL, ""})
    return sorted(set.intersection(*sets))


def fetch_line(name: str, root: Path, panel: list[str], max_cells: int = 32,
               n_ctrl: int = 3000, seed: int = 0, threads: int = 24) -> None:
    """Stage 2: select the stratified subsample for one line and fetch its rows."""
    d = root / name
    obs = pd.read_parquet(d / "obs.parquet")
    obs["gene"] = obs["gene"].astype(str)
    layout = Layout(**json.loads((d / "layout.json").read_text()))
    rows = select_cells(obs, set(panel), max_cells, n_ctrl, seed)
    np.save(d / "selected_rows.npy", rows)
    print(f"{name}: {rows.size} cells selected, {rows.size * layout.n_genes * 4 / 1e9:.2f} GB", flush=True)
    fetch_rows(SOURCES[name], layout, rows, d / "shards", threads=threads)


# ----------------------------------------------------------------------------- Arc HF source
HF_URL = ("https://huggingface.co/datasets/arcinstitute/State-Replogle-Filtered/resolve/main/"
          "replogle_concat.h5ad")


def hf_metadata(root: Path) -> None:
    """Read obs/var of the Arc Replogle-Nadig concatenation (log-normalised, KD-filtered)."""
    d = root / "hf"
    d.mkdir(parents=True, exist_ok=True)
    with RangeFile(HF_URL, root.parent / "hf_meta_blocks") as f, h5py.File(f, "r") as h:
        x = h["X"]
        assert x.chunks is None and x.compression is None and x.dtype == np.float32
        layout = Layout(int(x.id.get_offset()), int(x.shape[0]), int(x.shape[1]))

        def cat(name: str) -> np.ndarray:
            codes = h[f"obs/{name}/codes"][:]
            labels = _decode(h[f"obs/{name}/categories"][:])
            return labels[codes]

        obs = pd.DataFrame({
            "cell_line": cat("cell_line"), "gene": cat("gene"),
            "gem_group": h["obs/gem_group"][:], "UMI_count": h["obs/UMI_count"][:],
            "mitopercent": h["obs/mitopercent"][:],
        })
        var = pd.DataFrame({"gene_name": _decode(h["var/gene_name_index"][:])})
        for col in ("highly_variable", "means", "dispersions_norm", "in_matrix"):
            if col in h["var"]:
                var[col] = h[f"var/{col}"][:]
    obs.to_parquet(d / "obs.parquet")
    var.to_parquet(d / "var.parquet")
    (d / "layout.json").write_text(json.dumps(layout.__dict__))


def fetch_rows_lognorm(url: str, layout: Layout, rows: np.ndarray, out_dir: Path,
                       threads: int = 24, shard_rows: int = 1000, part: int = 0, parts: int = 1,
                       max_gap: int = 3, prefix: str = "shard") -> None:
    """Like ``fetch_rows`` but for log-normalised float rows (stored as sparse float16)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    row_bytes = layout.n_genes * 4
    runs = _ranges(rows, max_gap=max_gap, max_rows=96)
    shards: list[list[tuple[int, int]]] = [[]]
    count = 0
    for run in runs:
        shards[-1].append(run)
        count += run[1] - run[0]
        if count >= shard_rows:
            shards.append([])
            count = 0
    shards = [s for s in shards if s]
    rowset = set(rows.tolist())
    local = threading.local()

    def session() -> requests.Session:
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s  # type: ignore[no-any-return]

    def do_run(run: tuple[int, int]) -> tuple[np.ndarray, sp.csr_matrix]:
        lo = layout.offset + run[0] * row_bytes
        hi = layout.offset + run[1] * row_bytes
        buf = np.frombuffer(_fetch(url, lo, hi, session()), dtype=np.float32)
        mat = buf.reshape(run[1] - run[0], layout.n_genes)
        idx = np.arange(run[0], run[1])
        mask = np.array([i in rowset for i in idx])
        dense = mat[mask]
        assert np.all(np.isfinite(dense)) and dense.min() >= 0 and dense.max() < 20, "not log1p data"
        return idx[mask], sp.csr_matrix(dense)

    t0, done_bytes = time.time(), 0
    with cf.ThreadPoolExecutor(threads) as pool:
        for si, shard in enumerate(shards):
            path = out_dir / f"{prefix}_{si:05d}.npz"
            if path.exists() or si % parts != part:
                continue
            results = list(pool.map(do_run, shard))
            ids = np.concatenate([r[0] for r in results])
            mat = sp.vstack([r[1] for r in results]).tocsr()
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, rows=ids, data=mat.data.astype(np.float16),
                                indices=mat.indices.astype(np.int16),
                                indptr=mat.indptr, shape=np.array(mat.shape))
            tmp.replace(path)
            done_bytes += sum(r[1] - r[0] for r in shard) * row_bytes
            rate = done_bytes / max(time.time() - t0, 1e-9) / 1e6
            print(f"hf shard {si + 1}/{len(shards)} {rate:.2f} MB/s", flush=True)


def hf_fetch_remaining(root: Path, threads: int = 24, part: int = 0, parts: int = 1) -> None:
    """Fetch selected rows not yet present in any shard, as exact-row requests (no gap
    merging, so no wasted bytes). Writes ``shardB_*`` files next to the first-pass shards."""
    d = root / "hf"
    layout = Layout(**json.loads((d / "layout.json").read_text()))
    wanted = np.load(d / "selected_rows.npy")
    marker = d / "remaining_rows.npy"
    if not marker.exists():
        have = [np.load(p)["rows"] for p in sorted((d / "shards").glob("shard_*.npz")) if ".tmp" not in p.name]
        todo = np.setdiff1d(wanted, np.concatenate(have)) if have else wanted
        np.save(marker, todo)  # frozen once so that all workers agree on the shard layout
    todo = np.load(marker)
    print(f"remaining rows: {todo.size}", flush=True)
    fetch_rows_lognorm(HF_URL, layout, todo, d / "shards", threads=threads, part=part, parts=parts,
                       max_gap=1, prefix="shardB")


def hf_fetch(root: Path, max_cells: int = 32, n_ctrl: int = 3000, seed: int = 0,
             threads: int = 24, part: int = 0, parts: int = 1) -> None:
    """Select the stratified subsample in every line of the HF file and fetch it."""
    d = root / "hf"
    obs = pd.read_parquet(d / "obs.parquet")
    layout = Layout(**json.loads((d / "layout.json").read_text()))
    rows = []
    for line, g in obs.groupby("cell_line"):
        # select_cells returns positions within g; map them back to file rows
        pos = select_cells(g.reset_index(drop=True), set(g["gene"]) - {CONTROL_LABEL},
                           max_cells, n_ctrl, seed)
        rows.append(g.index.to_numpy()[pos])
        print(f"{line}: {len(rows[-1])} cells selected", flush=True)
    allrows = np.sort(np.concatenate(rows))
    np.save(d / "selected_rows.npy", allrows)
    print(f"total {allrows.size} cells, {allrows.size * layout.n_genes * 4 / 1e9:.2f} GB", flush=True)
    fetch_rows_lognorm(HF_URL, layout, allrows, d / "shards", threads=threads, part=part, parts=parts)
