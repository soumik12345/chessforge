"""Grain input pipelines (train: infinite shuffled; eval: finite, padded, multi-host safe)."""
from __future__ import annotations

import glob as globlib
import math
import os
import shutil

import grain.python as pygrain
import jax
import numpy as np
from etils import epath
from jax.experimental import multihost_utils

from .. import chess_utils as cu
from . import records


def list_files(pattern: str) -> list[str]:
    if not pattern:
        return []
    if pattern.startswith("gs://"):
        star = pattern.find("*")
        base = pattern[: pattern.rfind("/", 0, star)] if star >= 0 else os.path.dirname(pattern)
        rest = pattern[len(base) + 1:]
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
        local = os.path.join(cache_dir, f[len("gs://"):].replace("/", "__"))
        if not os.path.exists(local):
            tmp = local + ".tmp"
            with epath.Path(f).open("rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 << 20)
            os.replace(tmp, local)
        out.append(local)
    return out


def empty_batch(batch_size: int, history: int, with_legal: bool = True) -> dict[str, np.ndarray]:
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
    """bytes -> dict of numpy arrays. Board one-hot happens on device (int8 over PCIe)."""

    def __init__(self, history: int, history_mask_prob: float = 0.0, with_legal: bool = False):
        self.history = history
        self.history_mask_prob = history_mask_prob
        self.with_legal = with_legal

    def random_map(self, raw: bytes, rng: np.random.Generator):
        rec, legal = records.decode_position(raw)
        stored = rec["boards"].shape[0]
        if self.history > stored:
            raise ValueError(f"Model history {self.history} > stored history {stored}; re-run preprocessing")
        boards = rec["boards"][stored - self.history:]
        n_hist = min(int(rec["n_hist"]), self.history - 1)
        if self.history_mask_prob > 0 and rng.random() < self.history_mask_prob:
            n_hist = int(rng.integers(0, n_hist + 1))
        rel = cu.to_relative_codes(boards, int(rec["stm"]), n_hist)
        self_elo, oppo_elo = float(rec["self_elo"]), float(rec["oppo_elo"])
        time_info = np.array([rec["base_time"] / 100.0, rec["increment"] / 100.0, rec["clk_left"] / 100.0],
                             dtype=np.float32)
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
    ds = pygrain.MapDataset.source(pygrain.ArrayRecordDataSource(files))
    if not shard_by_file:
        ds = ds[pidx::pcount]
    return ds


def make_train_iterator(data_cfg, history: int, *, pattern: str | None = None, seed: int | None = None):
    seed = data_cfg.shuffle_seed if seed is None else seed
    pcount, pidx = jax.process_count(), jax.process_index()
    assert data_cfg.global_batch_size % pcount == 0
    per_host = data_cfg.global_batch_size // pcount
    ds = _host_dataset(pattern or data_cfg.train_glob, data_cfg.local_cache_dir)
    n_local = len(ds)
    ds = ds.shuffle(seed=seed + 7919 * pidx).repeat()
    ds = ds.random_map(FeaturizeTransform(history, data_cfg.history_mask_prob, with_legal=True),
                       seed=seed + 104729 * (pidx + 1))
    it_ds = ds.to_iter_dataset(pygrain.ReadOptions(num_threads=data_cfg.read_threads,
                                                   prefetch_buffer_size=data_cfg.prefetch_buffer_size))
    it_ds = it_ds.batch(per_host, drop_remainder=True)
    if data_cfg.num_workers > 0:
        it_ds = it_ds.mp_prefetch(pygrain.MultiprocessingOptions(num_workers=data_cfg.num_workers))
    return iter(it_ds), n_local


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


def eval_batches(pattern: str, history: int, global_batch_size: int, cache_dir: str = "",
                 max_batches: int | None = None, read_threads: int = 16):
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
    it = iter(ds.to_iter_dataset(pygrain.ReadOptions(num_threads=read_threads, prefetch_buffer_size=64))
              .batch(per_host, drop_remainder=False)) if n_local > 0 else iter(())
    empty = empty_batch(per_host, history, with_legal=True)
    for _ in range(n_batches):
        b = next(it, None)
        yield empty if b is None else _pad(b, per_host)