"""High-throughput input pipelines supporting both FORMAT_VERSION 2 and 3 shards.

FastShardIterator uses exact 1-pass per-shard sampling without replacement across
an in-RAM multi-shard buffer, eliminating short-horizon duplicates (0% repeats
within an epoch, 100% shard coverage) while mixing across active shards every batch.
"""

from __future__ import annotations

import concurrent.futures
import glob as globlib
import math
import os
import shutil
from collections import deque

import grain.python as pygrain
import jax
import numpy as np
from etils import epath
from jax.experimental import multihost_utils

from .. import chess_utils as cu
from . import records

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


def _build_rich_time_and_meta(
    base_s: np.ndarray,
    inc_s: np.ndarray,
    clk_left: np.ndarray,
    opp_clk_left: np.ndarray,
    prev_opp_ponder: np.ndarray,
    has_clock: np.ndarray,
    stm: np.ndarray,
    castling: np.ndarray,
    ep_file: np.ndarray,
    rep_count: np.ndarray,
):
    l_base = np.log1p(np.maximum(base_s, 0.0)) / 8.0
    l_inc = np.log1p(np.maximum(inc_s, 0.0)) / 4.0
    l_self = np.log1p(np.maximum(clk_left, 0.0)) / 8.0
    l_opp = np.log1p(np.maximum(opp_clk_left, 0.0)) / 8.0
    prev_p = np.where(np.isfinite(prev_opp_ponder), np.maximum(prev_opp_ponder, 0.0), 0.0)
    l_prev = np.log1p(prev_p) / 5.0
    rich_time = np.stack(
        [l_base, l_inc, l_self, l_opp, (l_self - l_opp) * 2.0, l_prev], axis=-1
    ).astype(np.float32)
    rich_time = rich_time * has_clock[:, None]

    board_meta = np.stack(
        [
            stm.astype(np.float32),
            ((castling & 1) != 0).astype(np.float32),
            ((castling & 2) != 0).astype(np.float32),
            ((castling & 4) != 0).astype(np.float32),
            ((castling & 8) != 0).astype(np.float32),
            (ep_file >= 0).astype(np.float32),
            (rep_count >= 1).astype(np.float32),
            (rep_count >= 2).astype(np.float32),
        ],
        axis=-1,
    )
    ep_sq = np.where(ep_file >= 0, 40 + ep_file.astype(np.int32), -1).astype(np.int32)
    return rich_time, board_meta, ep_sq


def featurize_records_batch(
    rec: np.ndarray,
    legal_idx: np.ndarray,
    history: int,
    stored_history: int,
    history_mask_prob: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Vectorized featurization over a (B,) structured array of v3 position_dtype."""
    B = rec.shape[0]
    H = history
    boards = rec["boards"][:, stored_history - H :]
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
    base_s = rec["base_time"].astype(np.float32)
    inc_s = rec["increment"].astype(np.float32)
    clk_left = rec["clk_left"].astype(np.float32)
    opp_clk_left = rec["opp_clk_left"].astype(np.float32)
    has_clock = rec["has_clock"].astype(np.float32)

    time_info = np.stack([base_s / 100.0, inc_s / 100.0, clk_left / 100.0], axis=-1)
    rich_time, board_meta, ep_sq = _build_rich_time_and_meta(
        base_s,
        inc_s,
        clk_left,
        opp_clk_left,
        rec["prev_opp_ponder"].astype(np.float32),
        has_clock,
        stm,
        rec["castling"].astype(np.uint8),
        rec["ep_file"].astype(np.int32),
        rec["rep_count"].astype(np.int32),
    )

    legal = np.zeros((B, cu.VOCAB_SIZE), dtype=np.bool_)
    np.put_along_axis(legal, legal_idx.astype(np.intp), True, axis=1)
    no_legal = rec["n_legal"] <= 0
    if no_legal.any():
        legal[no_legal] = True

    return {
        "board": rel.astype(np.int8),
        "time_info": time_info,
        "rich_time": rich_time,
        "has_clock": has_clock,
        "board_meta": board_meta,
        "ep_square_idx": ep_sq,
        "self_title": rec["self_title"].astype(np.int32),
        "oppo_title": rec["oppo_title"].astype(np.int32),
        "game_format": rec["game_format"].astype(np.int32),
        "self_elo": self_elo,
        "oppo_elo": oppo_elo,
        "game_elo": (self_elo + oppo_elo) * np.float32(0.5),
        "move": rec["move"].astype(np.int32),
        "next_move": rec["next_move"].astype(np.int32),
        "sf_move": rec["sf_move"].astype(np.int32),
        "sf_cp_loss": rec["sf_cp_loss"].astype(np.float32),
        "value": rec["value"].astype(np.int32),
        "ponder": rec["ponder"].astype(np.float32),
        "clk_left": clk_left,
        "weight": np.ones((B,), dtype=np.float32),
        "legal": legal,
    }


class FastShardIterator:
    """Exact 1-pass multi-shard streaming iterator via C++ ArrayRecordReader.read_all().

    Each active shard in RAM is shuffled once and consumed strictly without replacement
    until exhausted, then swapped for another shard from the same epoch if available.
    A batch can span epochs, with the current epoch's remaining records emitted first.
    """

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
        from array_record.python.array_record_module import ArrayRecordReader

        self._reader_cls = ArrayRecordReader
        self.base_files = list(files)
        self.batch_size = batch_size
        self.history = history
        self.stored_history = history
        self.history_mask_prob = history_mask_prob
        self.seed = seed
        self.dt = records.position_dtype(self.history, records.FORMAT_VERSION)

        self.rng = np.random.default_rng(seed)
        self.feat_rng = np.random.default_rng(seed + 104729)
        self.buffer_shards = min(buffer_shards, len(self.base_files))
        if self.buffer_shards < 1:
            raise ValueError(
                "FastShardIterator requires files and at least one buffer slot."
            )

        self._cache = {}
        self._pending = deque()
        self._loader = None
        if self.buffer_shards < len(self.base_files):
            self._loader = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._start_epoch(0)

    def _start_epoch(
        self, epoch: int, *, file_idx=None, slot_indices=None, positions=None
    ):
        for future in self._pending:
            future.cancel()
        self._pending.clear()
        self.epoch = epoch
        self.files = list(self.base_files)
        np.random.default_rng([self.seed, epoch]).shuffle(self.files)
        if slot_indices is None:
            slot_indices = list(range(self.buffer_shards))
        paths = [self.files[i] for i in slot_indices]
        if self._cache:
            loaded = [self._cache[path] for path in paths]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.buffer_shards
            ) as ex:
                loaded = list(ex.map(self._load_shard, paths))
            if self._loader is None:
                self._cache = dict(zip(paths, loaded))

        self.file_idx = self.buffer_shards if file_idx is None else file_idx
        self._slot_indices = list(slot_indices)
        self._slots = [arr for arr, _ in loaded]
        self._legal_slots = [leg for _, leg in loaded]
        self._slot_perms = [
            self._permutation(i, len(arr)) for i, (arr, _) in zip(slot_indices, loaded)
        ]
        self._slot_pos = list(positions) if positions is not None else [0] * len(loaded)
        self._prefetch()
        for slot, arr in enumerate(self._slots):
            if not len(arr):
                self._refill_slot(slot)
        if not any(len(arr) for arr in self._slots):
            raise RuntimeError("No valid records found in training shards.")

    def _permutation(self, file_idx: int, size: int):
        # A separate stream per shard makes partially consumed orders reproducible.
        seed = np.random.SeedSequence(self.seed, spawn_key=(self.epoch, file_idx))
        return np.random.default_rng(seed).permutation(size)

    def _prefetch(self):
        if self._loader is None:
            return
        first = self.file_idx + len(self._pending)
        stop = min(len(self.files), self.file_idx + self.buffer_shards)
        for i in range(first, stop):
            self._pending.append(self._loader.submit(self._load_shard, self.files[i]))

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
                r.read_all()
                if hasattr(r, "read_all")
                else [r.read() for _ in range(n)]
            )
        finally:
            r.close()

        ver = int(recs[0][0])
        if ver not in (records.LEGACY_FORMAT_VERSION, records.FORMAT_VERSION):
            raise ValueError(f"Unsupported record version {ver} in {path}")
        shard_h = int(recs[0][1])
        if shard_h < self.history:
            raise ValueError(
                f"Model history {self.history} > shard stored history {shard_h} in {path}"
            )
        raw_dt = records.position_dtype(shard_h, ver)
        nbytes = raw_dt.itemsize

        buf = b"".join(raw[2 : 2 + nbytes] for raw in recs)
        arr = np.frombuffer(buf, dtype=raw_dt).copy()
        if ver == records.LEGACY_FORMAT_VERSION:
            arr = records.upgrade_v2_to_v3(arr, shard_h)

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

    def _refill_slot(self, s: int):
        """Leave the slot exhausted when no more shards remain in this epoch."""
        while self._pending:
            arr, leg = self._pending[0].result()
            self._pending.popleft()
            i = self.file_idx
            self.file_idx += 1
            self._prefetch()
            if len(arr):
                self._slots[s], self._legal_slots[s] = arr, leg
                self._slot_indices[s] = i
                self._slot_perms[s] = self._permutation(i, len(arr))
                self._slot_pos[s] = 0
                return

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        need = self.batch_size
        k_slots = len(self._slots)
        rec_parts: list[np.ndarray] = []
        leg_parts: list[np.ndarray] = []

        epoch_starts = [0]
        while need > 0:
            rem = np.array(
                [len(self._slot_perms[s]) - self._slot_pos[s] for s in range(k_slots)],
                dtype=np.int64,
            )
            total_rem = int(rem.sum())
            if total_rem <= 0:
                self._start_epoch(self.epoch + 1)
                drawn = self.batch_size - need
                if drawn:
                    epoch_starts.append(drawn)
                continue

            if k_slots == 1:
                take = np.array([min(need, int(rem[0]))], dtype=np.int64)
            else:
                probs = rem / float(total_rem)
                counts = self.rng.multinomial(need, probs)
                take = np.minimum(counts, rem)
                if int(take.sum()) == 0:
                    s_max = int(np.argmax(rem))
                    take[s_max] = min(need, int(rem[s_max]))

            for s in range(k_slots):
                t = int(take[s])
                if t <= 0:
                    continue
                p0 = self._slot_pos[s]
                p1 = p0 + t
                idx = self._slot_perms[s][p0:p1]
                rec_parts.append(self._slots[s][idx])
                leg_parts.append(self._legal_slots[s][idx])
                self._slot_pos[s] = p1
                if p1 >= len(self._slot_perms[s]):
                    self._refill_slot(s)
            need -= int(take.sum())

        if len(rec_parts) == 1:
            batch_rec = rec_parts[0]
            batch_leg = leg_parts[0]
        else:
            batch_rec = np.concatenate(rec_parts, axis=0)
            batch_leg = np.concatenate(leg_parts, axis=0)
            # Preserve epoch boundaries in the returned record order.
            perm = np.arange(self.batch_size)
            for start, stop in zip(epoch_starts, epoch_starts[1:] + [self.batch_size]):
                self.rng.shuffle(perm[start:stop])
            batch_rec = batch_rec[perm]
            batch_leg = batch_leg[perm]

        return featurize_records_batch(
            batch_rec,
            batch_leg,
            self.history,
            self.stored_history,
            self.history_mask_prob,
            self.feat_rng,
        )

    def get_state(self) -> dict:
        return {
            "epoch": self.epoch,
            "file_idx": self.file_idx,
            "slot_indices": self._slot_indices.copy(),
            "positions": self._slot_pos.copy(),
            "rng": self.rng.bit_generator.state,
            "feat_rng": self.feat_rng.bit_generator.state,
        }

    def set_state(self, state: dict):
        epoch = int(state.get("epoch", 0))
        if "slot_indices" in state:
            if len(state["slot_indices"]) != self.buffer_shards:
                raise ValueError(
                    "Restoring active shard cursors requires the same buffer size."
                )
            self._start_epoch(
                epoch,
                file_idx=state["file_idx"],
                slot_indices=state["slot_indices"],
                positions=state["positions"],
            )
            self.rng.bit_generator.state = state["rng"]
            self.feat_rng.bit_generator.state = state["feat_rng"]
        else:
            # Legacy checkpoints only tracked the producer's file cursor.
            start = max(
                0,
                min(int(state.get("file_idx", 0)), len(self.files))
                - self.buffer_shards,
            )
            stop = start + self.buffer_shards
            self._start_epoch(
                epoch, file_idx=stop, slot_indices=list(range(start, stop))
            )


def empty_batch(
    batch_size: int, history: int, with_legal: bool = True
) -> dict[str, np.ndarray]:
    out = {
        "board": np.zeros((batch_size, history, 64), np.int8),
        "time_info": np.zeros((batch_size, 3), np.float32),
        "rich_time": np.zeros((batch_size, 6), np.float32),
        "has_clock": np.zeros((batch_size,), np.float32),
        "board_meta": np.zeros((batch_size, 8), np.float32),
        "ep_square_idx": np.full((batch_size,), -1, np.int32),
        "self_title": np.zeros((batch_size,), np.int32),
        "oppo_title": np.zeros((batch_size,), np.int32),
        "game_format": np.zeros((batch_size,), np.int32),
        "self_elo": np.zeros((batch_size,), np.float32),
        "oppo_elo": np.zeros((batch_size,), np.float32),
        "game_elo": np.zeros((batch_size,), np.float32),
        "move": np.zeros((batch_size,), np.int32),
        "next_move": np.full((batch_size,), -1, np.int32),
        "sf_move": np.full((batch_size,), -1, np.int32),
        "sf_cp_loss": np.full((batch_size,), -1.0, np.float32),
        "value": np.zeros((batch_size,), np.int32),
        "ponder": np.zeros((batch_size,), np.float32),
        "clk_left": np.zeros((batch_size,), np.float32),
        "weight": np.zeros((batch_size,), np.float32),
    }
    if with_legal:
        out["legal"] = np.ones((batch_size, cu.VOCAB_SIZE), np.bool_)
    return out


class FeaturizeTransform(pygrain.RandomMapTransform):
    """bytes -> dict of numpy arrays (used for Grain eval)."""

    def __init__(
        self, history: int, history_mask_prob: float = 0.0, with_legal: bool = False
    ):
        self.history = history
        self.history_mask_prob = history_mask_prob
        self.with_legal = with_legal

    def random_map(self, raw: bytes, rng: np.random.Generator):
        rec, legal = records.decode_position(raw)
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
        base_s = float(rec["base_time"])
        inc_s = float(rec["increment"])
        clk_left = float(rec["clk_left"])
        opp_clk_left = float(rec["opp_clk_left"])
        has_clock = float(rec["has_clock"])

        rich_time, board_meta, ep_sq = _build_rich_time_and_meta(
            np.asarray([base_s], np.float32),
            np.asarray([inc_s], np.float32),
            np.asarray([clk_left], np.float32),
            np.asarray([opp_clk_left], np.float32),
            np.asarray([rec["prev_opp_ponder"]], np.float32),
            np.asarray([has_clock], np.float32),
            np.asarray([rec["stm"]], np.int32),
            np.asarray([rec["castling"]], np.uint8),
            np.asarray([rec["ep_file"]], np.int32),
            np.asarray([rec["rep_count"]], np.int32),
        )
        out = {
            "board": rel.astype(np.int8),
            "time_info": np.array(
                [base_s / 100.0, inc_s / 100.0, clk_left / 100.0], dtype=np.float32
            ),
            "rich_time": rich_time[0],
            "has_clock": np.float32(has_clock),
            "board_meta": board_meta[0],
            "ep_square_idx": np.int32(ep_sq[0]),
            "self_title": np.int32(rec["self_title"]),
            "oppo_title": np.int32(rec["oppo_title"]),
            "game_format": np.int32(rec["game_format"]),
            "self_elo": np.float32(self_elo),
            "oppo_elo": np.float32(oppo_elo),
            "game_elo": np.float32((self_elo + oppo_elo) * 0.5),
            "move": np.int32(rec["move"]),
            "next_move": np.int32(rec["next_move"]),
            "sf_move": np.int32(rec["sf_move"]),
            "sf_cp_loss": np.float32(rec["sf_cp_loss"]),
            "value": np.int32(rec["value"]),
            "ponder": np.float32(rec["ponder"]),
            "clk_left": np.float32(clk_left),
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
    pad_defaults = {
        "legal": True,
        "ep_square_idx": -1,
        "next_move": -1,
        "sf_move": -1,
        "sf_cp_loss": -1.0,
    }
    out = {}
    for k, v in batch.items():
        pad_val = pad_defaults.get(k, 0)
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
