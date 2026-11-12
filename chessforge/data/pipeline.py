"""High-throughput input pipelines (train: sequential C++ shard reader + RAM shuffle pool; eval: Grain)."""

from __future__ import annotations

import concurrent.futures
import glob as globlib
import math
import os
import queue
import shutil
import threading
from typing import cast

import grain.python as pygrain
import jax
import numpy as np
from etils import epath
from grain.sources import RandomAccessDataSource
from jax.experimental import multihost_utils

from chessforge import chess_utils as cu
from chessforge.data import records

# Maximum number of legal moves in any reachable chess position is 218; 224 aligns to 32B.
MAX_LEGAL_MOVES = 224


def list_files(pattern: str) -> list[str]:
    if not pattern:
        return []
    if pattern.startswith("gs://"):
        star = pattern.find("*")
        base = (
            pattern[: pattern.rfind("/", 0, star)]
            if star >= 0
            else os.path.dirname(pattern)
        )
        rest = pattern[len(base) + 1 :]
        return sorted(str(p) for p in epath.Path(base).glob(rest))
    return sorted(globlib.glob(pattern))


def stage_files(files: list[str], cache_dir: str) -> list[str]:
    if not cache_dir:
        return files
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for f in files:
        if not f.startswith("gs://"):
            out.append(f)
            continue
        local = os.path.join(cache_dir, f[len("gs://") :].replace("/", "__"))
        if not os.path.exists(local):
            tmp = local + ".tmp"
            with epath.Path(f).open("rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 << 20)
            os.replace(tmp, local)
        out.append(local)
    return out


def featurize_records_batch(
    rec: np.ndarray,
    legal_idx: np.ndarray,
    history: int,
    stored_history: int,
    history_mask_prob: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Vectorized featurization over a (B,) structured array of position_dtype."""
    B = rec.shape[0]
    H = history
    boards = rec["boards"][:, stored_history - H :]  # (B, H, 64)
    n_hist = np.minimum(rec["n_hist"].astype(np.int32), H - 1)
    if history_mask_prob > 0:
        mask_draw = rng.random(B) < history_mask_prob
        if mask_draw.any():
            n_hist = np.where(mask_draw, rng.integers(0, n_hist + 1), n_hist)

    stm = rec["stm"].astype(np.int32)
    lag = (H - 1) - np.arange(H, dtype=np.int32)[None, :]
    black = (stm[:, None] ^ (lag & 1)) == 1
    rel = boards.copy()
    if black.any():
        rel[black] = cu.SWAP_COLOR[boards[black][:, cu.MIRROR_SQ]]

    first = (H - 1) - n_hist
    idx = np.maximum(np.arange(H, dtype=np.int32)[None, :], first[:, None])
    rel = np.take_along_axis(rel, idx[:, :, None], axis=1)

    self_elo = rec["self_elo"].astype(np.float32)
    oppo_elo = rec["oppo_elo"].astype(np.float32)
    time_info = np.stack(
        [
            rec["base_time"].astype(np.float32) / 100.0,
            rec["increment"].astype(np.float32) / 100.0,
            rec["clk_left"].astype(np.float32) / 100.0,
        ],
        axis=-1,
    )

    legal = np.zeros((B, cu.VOCAB_SIZE), dtype=np.bool_)
    np.put_along_axis(legal, legal_idx.astype(np.intp), True, axis=1)
    no_legal = rec["n_legal"] <= 0
    if no_legal.any():
        legal[no_legal] = True

    return {
        "board": rel.astype(np.int8),
        "time_info": time_info,
        "self_elo": self_elo,
        "oppo_elo": oppo_elo,
        "game_elo": (self_elo + oppo_elo) * np.float32(0.5),
        "move": rec["move"].astype(np.int32),
        "value": rec["value"].astype(np.int32),
        "ponder": rec["ponder"].astype(np.float32),
        "weight": np.ones((B,), dtype=np.float32),
        "legal": legal,
    }


class FastShardIterator:
    """Streams whole ArrayRecord shards sequentially via C++ read_all() into an in-RAM pool."""

    def __init__(
        self,
        files: list[str],
        batch_size: int,
        history: int,
        history_mask_prob: float = 0.0,
        seed: int = 0,
        buffer_shards: int = 8,
        stored_history: int | None = None,
    ):
        from array_record.python.array_record_module import ArrayRecordReader  # ty: ignore[import-not-found]

        self._reader_cls = ArrayRecordReader
        self.base_files = list(files)
        self.batch_size = batch_size
        self.history = history
        self.stored_history = history
        self.history_mask_prob = history_mask_prob
        self.seed = seed
        self.dt = records.position_dtype(self.history)

        self._lock = threading.Lock()
        self.epoch = 0
        self.file_idx = 0
        self.rng = np.random.default_rng(seed)
        self.feat_rng = np.random.default_rng(seed + 104729)
        self._shuffle_files_locked()

        self.buffer_shards = min(buffer_shards, len(self.files))
        self._q: queue.Queue[tuple[np.ndarray, np.ndarray]] = queue.Queue(maxsize=4)
        self._load_initial_slots()

        self._bg = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._bg.start()

    def _shuffle_files_locked(self):
        self.files = list(self.base_files)
        np.random.default_rng([self.seed, self.epoch]).shuffle(self.files)

    def _next_path(self) -> str:
        with self._lock:
            return self._next_path_locked()

    def _next_path_locked(self) -> str:
        if self.file_idx >= len(self.files):
            self.epoch += 1
            self.file_idx = 0
            self._shuffle_files_locked()
        p = self.files[self.file_idx]
        self.file_idx += 1
        return p

    def _load_initial_slots(self):
        with self._lock:
            init_paths = [self._next_path_locked() for _ in range(self.buffer_shards)]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.buffer_shards)
        ) as ex:
            loaded = list(ex.map(self._load_shard, init_paths))
        valid = [(arr, leg) for arr, leg in loaded if arr.size > 0]
        if not valid:
            raise RuntimeError("No valid records found in training shards.")
        self._slots = [arr for arr, _ in valid]
        self._legal_slots = [leg for _, leg in valid]
        self._replace_slot = 0
        self._rebuild_pool()

    def _load_shard(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        r = self._reader_cls(path)
        try:
            n = r.num_records()
            if n == 0:
                return (
                    np.zeros(0, dtype=self.dt),
                    np.zeros((0, MAX_LEGAL_MOVES), dtype=np.int16),
                )
            recs = (
                r.read_all() if hasattr(r, "read_all") else [r.read() for _ in range(n)]
            )
        finally:
            r.close()

        if recs[0][0] != records.FORMAT_VERSION:
            raise ValueError(f"Unsupported record version {recs[0][0]} in {path}")
        shard_h = int(recs[0][1])
        if shard_h < self.history:
            raise ValueError(
                f"Model history {self.history} > shard stored history {shard_h} in {path}"
            )
        shard_dt = records.position_dtype(shard_h)
        nbytes = shard_dt.itemsize

        buf = b"".join(raw[2 : 2 + nbytes] for raw in recs)
        arr = np.frombuffer(buf, dtype=shard_dt).copy()

        # Unpack trailing variable-length int16 legal move indices in pure vectorized NumPy
        legal_raw = b"".join(raw[2 + nbytes :] for raw in recs)
        legal_buf = np.frombuffer(legal_raw, dtype=np.int16)
        raw_counts = arr["n_legal"].astype(np.int32)
        counts = np.minimum(raw_counts, MAX_LEGAL_MOVES)
        offsets = np.cumsum(raw_counts) - raw_counts
        k = np.arange(MAX_LEGAL_MOVES, dtype=np.int32)[None, :]
        pos = offsets[:, None] + np.minimum(k, np.maximum(counts[:, None] - 1, 0))
        if legal_buf.size > 0:
            legal_idx = np.where(
                counts[:, None] > 0,
                legal_buf[np.clip(pos, 0, legal_buf.size - 1)],
                arr["move"][:, None],
            ).astype(np.int16)
        else:
            legal_idx = np.broadcast_to(
                arr["move"][:, None].astype(np.int16), (len(arr), MAX_LEGAL_MOVES)
            ).copy()

        if shard_h != self.history:
            trimmed = np.empty(len(arr), dtype=self.dt)
            assert self.dt.names is not None
            for name in self.dt.names:
                if name == "boards":
                    trimmed["boards"] = arr["boards"][:, shard_h - self.history :]
                else:
                    trimmed[name] = arr[name]
            arr = trimmed

        return arr, legal_idx

    def _rebuild_pool(self):
        self._pool = np.concatenate(self._slots, axis=0)
        self._legal_pool = np.concatenate(self._legal_slots, axis=0)
        self._perm = self.rng.permutation(len(self._pool))
        self._cursor = 0
        self._swap_after = max(
            len(self._pool) // max(1, len(self._slots)), self.batch_size
        )

    def _prefetch_worker(self):
        while True:
            try:
                arr, leg = self._load_shard(self._next_path())
                if arr.size > 0:
                    self._q.put((arr, leg))
            except Exception:  # noqa: BLE001
                continue

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        if self._cursor >= self._swap_after and len(self.base_files) > len(self._slots):
            try:
                new_arr, new_leg = self._q.get_nowait()
                self._slots[self._replace_slot] = new_arr
                self._legal_slots[self._replace_slot] = new_leg
                self._replace_slot = (self._replace_slot + 1) % len(self._slots)
                self._rebuild_pool()
            except queue.Empty:
                if self._cursor + self.batch_size > len(self._pool):
                    new_arr, new_leg = self._q.get()
                    self._slots[self._replace_slot] = new_arr
                    self._legal_slots[self._replace_slot] = new_leg
                    self._replace_slot = (self._replace_slot + 1) % len(self._slots)
                    self._rebuild_pool()

        if self._cursor + self.batch_size > len(self._pool):
            self._perm = self.rng.permutation(len(self._pool))
            self._cursor = 0

        if len(self._pool) < self.batch_size:
            idx = self.rng.choice(len(self._pool), size=self.batch_size, replace=True)
        else:
            idx = self._perm[self._cursor : self._cursor + self.batch_size]
            self._cursor += self.batch_size

        return featurize_records_batch(
            self._pool[idx],
            self._legal_pool[idx],
            self.history,
            self.stored_history,
            self.history_mask_prob,
            self.feat_rng,
        )

    def get_state(self) -> dict:
        with self._lock:
            return {"epoch": self.epoch, "file_idx": self.file_idx}

    def set_state(self, state: dict):
        with self._lock:
            self.epoch = int(state.get("epoch", 0))
            # Step back by buffer_shards so active pool shards are reconstructed at the saved position
            saved_idx = int(state.get("file_idx", 0))
            self.file_idx = max(0, saved_idx - self.buffer_shards)
            self._shuffle_files_locked()
            reload_paths = [self._next_path_locked() for _ in range(self.buffer_shards)]

        # Drain any stale Epoch-0 shards queued by the background worker before set_state
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.buffer_shards)
        ) as ex:
            loaded = list(ex.map(self._load_shard, reload_paths))
        valid = [(arr, leg) for arr, leg in loaded if arr.size > 0]
        if valid:
            self._slots = [arr for arr, _ in valid]
            self._legal_slots = [leg for _, leg in valid]
            self._replace_slot = 0
            self._rebuild_pool()


def empty_batch(
    batch_size: int, history: int, with_legal: bool = True
) -> dict[str, np.ndarray]:
    out = {
        "board": np.zeros((batch_size, history, 64), np.int8),
        "time_info": np.zeros((batch_size, 3), np.float32),
        "self_elo": np.zeros((batch_size,), np.float32),
        "oppo_elo": np.zeros((batch_size,), np.float32),
        "game_elo": np.zeros((batch_size,), np.float32),
        "move": np.zeros((batch_size,), np.int32),
        "value": np.zeros((batch_size,), np.int32),
        "ponder": np.zeros((batch_size,), np.float32),
        "weight": np.zeros((batch_size,), np.float32),
    }
    if with_legal:
        out["legal"] = np.ones((batch_size, cu.VOCAB_SIZE), np.bool_)
    return out


class FeaturizeTransform(pygrain.RandomMapTransform):
    """bytes -> dict of numpy arrays (used for eval)."""

    def __init__(
        self, history: int, history_mask_prob: float = 0.0, with_legal: bool = False
    ):
        self.history = history
        self.history_mask_prob = history_mask_prob
        self.with_legal = with_legal

    def random_map(self, element: bytes, rng: np.random.Generator):
        rec, legal = records.decode_position(element)
        stored = rec["boards"].shape[0]
        if self.history > stored:
            raise ValueError(
                f"Model history {self.history} > stored history {stored}; re-run preprocessing"
            )
        boards = rec["boards"][stored - self.history :]
        n_hist = min(int(rec["n_hist"]), self.history - 1)
        if self.history_mask_prob > 0 and rng.random() < self.history_mask_prob:
            n_hist = int(rng.integers(0, n_hist + 1))
        rel = cu.to_relative_codes(boards, int(rec["stm"]), n_hist)
        self_elo, oppo_elo = float(rec["self_elo"]), float(rec["oppo_elo"])
        time_info = np.array(
            [
                rec["base_time"] / 100.0,
                rec["increment"] / 100.0,
                rec["clk_left"] / 100.0,
            ],
            dtype=np.float32,
        )
        out = {
            "board": rel.astype(np.int8),
            "time_info": time_info,
            "self_elo": np.float32(self_elo),
            "oppo_elo": np.float32(oppo_elo),
            "game_elo": np.float32((self_elo + oppo_elo) / 2),
            "move": np.int32(rec["move"]),
            "value": np.int32(rec["value"]),
            "ponder": np.float32(rec["ponder"]),
            "weight": np.float32(1.0),
        }
        if self.with_legal:
            mask = np.zeros(cu.VOCAB_SIZE, np.bool_)
            if legal.size:
                mask[legal.astype(np.int64)] = True
            else:
                mask[:] = True
            out["legal"] = mask
        return out


def _host_dataset(pattern: str, cache_dir: str):
    files = list_files(pattern)
    if not files:
        raise FileNotFoundError(f"No files match {pattern!r}")
    pidx, pcount = jax.process_index(), jax.process_count()
    shard_by_file = len(files) >= pcount
    if shard_by_file:
        files = files[pidx::pcount]
    files = stage_files(files, cache_dir)
    source = cast(
        RandomAccessDataSource[bytes],
        pygrain.ArrayRecordDataSource(files),
    )
    ds = pygrain.MapDataset.source(source)
    if not shard_by_file:
        ds = ds[pidx::pcount]
    return ds


def make_train_iterator(
    data_cfg, history: int, *, pattern: str | None = None, seed: int | None = None
):
    seed = data_cfg.shuffle_seed if seed is None else seed
    pcount, pidx = jax.process_count(), jax.process_index()
    assert data_cfg.global_batch_size % pcount == 0
    per_host = data_cfg.global_batch_size // pcount
    files = list_files(pattern or data_cfg.train_glob)
    if not files:
        raise FileNotFoundError(f"No files match {pattern or data_cfg.train_glob!r}")
    if len(files) >= pcount:
        files = files[pidx::pcount]
    files = stage_files(files, data_cfg.local_cache_dir)
    it = FastShardIterator(
        files=files,
        batch_size=per_host,
        history=history,
        history_mask_prob=data_cfg.history_mask_prob,
        seed=seed + 7919 * pidx,
        buffer_shards=8,
    )
    if len(files) <= it.buffer_shards:
        n_local = int(sum(len(s) for s in it._slots))
    else:
        avg_per_shard = sum(len(s) for s in it._slots) / max(1, len(it._slots))
        n_local = int(round(avg_per_shard * len(files)))
    return it, n_local


def _pad(batch: dict, size: int) -> dict:
    n = batch["weight"].shape[0]
    if n == size:
        return batch
    out = {}
    for k, v in batch.items():
        pad_val = True if k == "legal" else 0
        pad = np.full((size - n,) + v.shape[1:], pad_val, dtype=v.dtype)
        out[k] = np.concatenate([v, pad], 0)
    return out


def eval_batches(
    pattern: str,
    history: int,
    global_batch_size: int,
    cache_dir: str = "",
    max_batches: int | None = None,
    read_threads: int = 16,
):
    """Yields per-host batches; all hosts yield the same number (zero-weight padding)."""
    pcount = jax.process_count()
    per_host = global_batch_size // pcount
    ds = _host_dataset(pattern, cache_dir)
    n_local = len(ds)
    ds = ds.random_map(FeaturizeTransform(history, 0.0, with_legal=True), seed=0)
    n_batches = math.ceil(n_local / per_host) if n_local > 0 else 0
    n_batches = int(np.max(multihost_utils.process_allgather(np.array(n_batches))))
    if max_batches:
        n_batches = min(n_batches, max_batches)
    it = (
        iter(
            ds.to_iter_dataset(
                pygrain.ReadOptions(num_threads=read_threads, prefetch_buffer_size=64)
            ).batch(per_host, drop_remainder=False)
        )
        if n_local > 0
        else iter(())
    )
    empty = empty_batch(per_host, history, with_legal=True)
    for _ in range(n_batches):
        b = next(it, None)
        yield empty if b is None else _pad(b, per_host)
