"""Human-bounded JAX UCI chess engine for ChessformerV2 & Maia-3 checkpoints.

Features:
  - Full v3 board state tracking (castling, EP, Zobrist repetitions, live UCI clocks)
  - Official title conditioning (GM, IM, FM, CM/NM, or Untitled) & format (Blitz/Rapid/OTB)
  - min_p truncation + ponder-effort adaptive temperature + 1-ply human WDL guardrail
"""

from __future__ import annotations

import argparse
import sys
from collections import deque

import chess
import jax
import numpy as np
from flax import nnx

from chessforge import chess_utils as cu
from chessforge.config import load_train_config
from chessforge.data.pipeline import _build_rich_time_and_meta
from chessforge.models import build_model, forward_model
from chessforge.train.trainer import Trainer


def _probs_to_permille(probs: list[float]) -> tuple[int, int, int]:
    scaled = [max(0.0, float(p)) * 1000.0 for p in probs]
    ints = [int(v) for v in scaled]
    rem = 1000 - sum(ints)
    order = sorted(range(3), key=lambda i: scaled[i] - ints[i], reverse=True)
    for idx in order[: max(0, rem)]:
        ints[idx] += 1
    return ints[0], ints[1], ints[2]


class ChessforgeUCIEngine:
    def __init__(self, args):
        self.args = args
        if args.config:
            tcfg = load_train_config(args.config, args.set)
            trainer = Trainer(tcfg)
            self.model_cfg = trainer.model_cfg
            self.graphdef = trainer.graphdef
            self.params = trainer.restore_params(args.step)
        else:
            from chessforge.models.convert_torch import load_released

            self.model_cfg, self.params = load_released(args.released, "float32")
            self.graphdef, _ = nnx.split(build_model(self.model_cfg, rngs=0))

        @jax.jit
        def _infer(params, batch):
            model = nnx.merge(self.graphdef, params)
            return forward_model(model, batch, deterministic=True, return_aux=True)

        self._infer = _infer
        self.rng = np.random.default_rng(args.seed)

        self.self_elo = float(args.elo)
        self.oppo_elo = float(args.elo)
        self.self_title = cu.parse_title_tier(args.title)
        self.oppo_title = cu.parse_title_tier(args.title)
        self.game_format = {"blitz": 0, "rapid": 1, "otb": 2}[args.format.lower()]
        self.temperature = float(args.temperature)
        self.min_p = float(args.min_p)
        self.max_wdl_drop = float(args.max_wdl_drop)
        self.effort_temp = bool(args.effort_adaptive_temp)
        self.multipv = max(1, min(20, int(args.multipv)))
        self.base_time = float(args.base_time)
        self.inc_time = float(args.inc_time)

        self.board = chess.Board()
        self.codes_hist: deque[np.ndarray] = deque(maxlen=self.model_cfg.history)
        self.seen_keys: dict[tuple, int] = {}
        self.pending_bestmove: chess.Move | None = None
        self.pending_search = False
        self._reset_game()

    def _reset_game(self):
        self.board = chess.Board()
        self.codes_hist.clear()
        self.seen_keys.clear()
        self.codes_hist.append(cu.board_codes(self.board))
        self.pending_bestmove = None
        self.pending_search = False

    def _make_single_batch(
        self,
        board: chess.Board,
        codes_seq: list[np.ndarray],
        rep_c: int,
        self_elo: float,
        oppo_elo: float,
        self_title: int,
        oppo_title: int,
        clk_self: float,
        clk_opp: float,
        has_clk: float,
    ) -> dict[str, np.ndarray]:
        H = self.model_cfg.history
        win = np.zeros((H, 64), np.int8)
        win[H - len(codes_seq) :] = np.stack(codes_seq[-H:])
        stm = int(board.turn == chess.BLACK)
        rel = cu.to_relative_codes(win, stm, len(codes_seq) - 1)
        cmask = cu.castling_mask(board)
        epf = cu.ep_file(board)
        rt, bm, ep_sq = _build_rich_time_and_meta(
            np.asarray([self.base_time], np.float32),
            np.asarray([self.inc_time], np.float32),
            np.asarray([clk_self], np.float32),
            np.asarray([clk_opp], np.float32),
            np.asarray([0.0], np.float32),
            np.asarray([has_clk], np.float32),
            np.asarray([stm], np.int32),
            np.asarray([cmask], np.uint8),
            np.asarray([epf], np.int32),
            np.asarray([rep_c], np.int32),
        )
        legal = np.zeros((1, cu.VOCAB_SIZE), dtype=bool)
        legal[0, cu.legal_move_indices(board).astype(np.int64)] = True
        return {
            "board": rel[None],
            "legal": legal,
            "time_info": np.asarray(
                [[self.base_time / 100.0, self.inc_time / 100.0, clk_self / 100.0]],
                np.float32,
            ),
            "rich_time": rt,
            "has_clock": np.asarray([has_clk], np.float32),
            "board_meta": bm,
            "ep_square_idx": ep_sq,
            "self_title": np.asarray([self_title], np.int32),
            "oppo_title": np.asarray([oppo_title], np.int32),
            "game_format": np.asarray([self.game_format], np.int32),
            "self_elo": np.asarray([self_elo], np.float32),
            "oppo_elo": np.asarray([oppo_elo], np.float32),
            "clk_left": np.asarray([clk_self], np.float32),
        }

    def _idx_to_legal_move(self, idx: int) -> chess.Move | None:
        uci = cu.index_to_uci(int(idx))
        if self.board.turn == chess.BLACK:
            uci = cu.mirror_uci(uci)
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return None
        return mv if mv in self.board.legal_moves else None

    def _parse_go_clocks(self, tokens: list[str]) -> tuple[float, float, float]:
        kv = {}
        for i in range(len(tokens) - 1):
            if tokens[i] in ("wtime", "btime", "winc", "binc"):
                try:
                    kv[tokens[i]] = float(tokens[i + 1]) / 1000.0
                except ValueError:
                    pass
        if not kv or self.game_format == 2:
            return 0.0, 0.0, 0.0
        if self.board.turn == chess.WHITE:
            s_clk = kv.get("wtime", self.base_time)
            o_clk = kv.get("btime", self.base_time)
            if "winc" in kv:
                self.inc_time = kv["winc"]
        else:
            s_clk = kv.get("btime", self.base_time)
            o_clk = kv.get("wtime", self.base_time)
            if "binc" in kv:
                self.inc_time = kv["binc"]
        return max(0.0, s_clk), max(0.0, o_clk), 1.0

    def search(self, line: str):
        if self.board.is_game_over():
            return None, []
        leg_idx = cu.legal_move_indices(self.board)
        if leg_idx.size == 0:
            return None, []

        s_clk, o_clk, has_clk = self._parse_go_clocks(line.split())
        rep_c = min(self.seen_keys.get(self.board._transposition_key(), 0), 2)
        batch = self._make_single_batch(
            self.board,
            list(self.codes_hist),
            rep_c,
            self.self_elo,
            self.oppo_elo,
            self.self_title,
            self.oppo_title,
            s_clk,
            o_clk,
            has_clk,
        )
        lm, _, _, aux = self._infer(self.params, batch)
        logits = np.asarray(lm[0], dtype=np.float64)
        effort = (
            float(np.asarray(aux["ponder_effort"][0]))
            if "ponder_effort" in aux
            else 0.25
        )

        mask = np.zeros(cu.VOCAB_SIZE, dtype=bool)
        mask[leg_idx.astype(np.int64)] = True
        masked = np.where(mask, logits, -np.inf)
        max_l = np.max(masked)
        raw_probs = np.where(mask, np.exp(masked - max_l), 0.0)
        raw_probs /= raw_probs.sum()

        # Evaluate top candidate moves for UCI MultiPV + 1-ply human WDL guardrail
        top_k = min(max(self.multipv, 8), int(leg_idx.size))
        cand_idxs = np.argsort(raw_probs)[::-1][:top_k]
        cands = []
        cand_batches = []
        for idx in cand_idxs:
            mv = self._idx_to_legal_move(int(idx))
            if mv is None:
                continue
            b_next = self.board.copy(stack=False)
            next_rep = (
                0
                if b_next.is_zeroing(mv)
                else min(self.seen_keys.get(b_next._transposition_key(), 0), 2)
            )
            b_next.push(mv)
            next_seq = [*self.codes_hist, cu.board_codes(b_next)]
            cand_batches.append(
                self._make_single_batch(
                    b_next,
                    next_seq,
                    next_rep,
                    self.oppo_elo,
                    self.self_elo,
                    self.oppo_title,
                    self.self_title,
                    o_clk,
                    s_clk,
                    has_clk,
                )
            )
            cands.append({"idx": int(idx), "move": mv, "prob": float(raw_probs[idx])})

        if cand_batches:
            stacked = {
                k: np.concatenate([cb[k] for cb in cand_batches], axis=0)
                for k in cand_batches[0]
            }
            _, cand_lv, _, _ = self._infer(self.params, stacked)
            cand_lv_np = np.asarray(cand_lv, dtype=np.float64)
            for item, v_logits in zip(cands, cand_lv_np):
                ev = np.exp(v_logits - np.max(v_logits))
                loss_opp, draw_opp, win_opp = (ev / ev.sum()).tolist()
                # Invert opponent WDL -> our (win, draw, loss)
                item["score_exp"] = loss_opp + 0.5 * draw_opp
                item["wdl"] = _probs_to_permille([loss_opp, draw_opp, win_opp])

        if not cands:
            return None, []

        # Human-bounded sampling: min_p truncation + 1-ply human WDL guardrail
        temp = self.temperature
        if temp > 0 and self.effort_temp:
            # Lower temperature on instant recaptures (low effort) & sharp tactical spikes
            temp = temp * float(
                np.clip(0.65 + 1.2 * effort * (1.0 - effort), 0.5, 1.05)
            )

        if temp <= 0.0 or len(cands) == 1:
            chosen = cands[0]["move"]
        else:
            best_p = cands[0]["prob"]
            best_exp = cands[0]["score_exp"]
            viable = [
                c
                for c in cands
                if c["prob"] >= self.min_p * best_p
                and (best_exp - c["score_exp"]) <= self.max_wdl_drop
            ] or [cands[0]]
            v_logits = np.array(
                [logits[c["idx"]] / temp for c in viable], dtype=np.float64
            )
            vp = np.exp(v_logits - np.max(v_logits))
            vp /= vp.sum()
            chosen = viable[int(self.rng.choice(len(viable), p=vp))]["move"]

        return chosen, cands[: self.multipv]

    def cmd_position(self, line: str):
        tokens = line.split()
        if len(tokens) < 2:
            return
        i = 1
        if tokens[i] == "startpos":
            board = chess.Board()
            i += 1
        elif tokens[i] == "fen" and len(tokens) >= i + 7:
            try:
                board = chess.Board(" ".join(tokens[i + 1 : i + 7]))
            except ValueError:
                return
            i += 7
        else:
            return

        moves = tokens[i + 1 :] if (i < len(tokens) and tokens[i] == "moves") else []
        seq: deque[np.ndarray] = deque(maxlen=self.model_cfg.history)
        seen: dict[tuple, int] = {}
        seq.append(cu.board_codes(board))
        for m_str in moves:
            try:
                mv = chess.Move.from_uci(m_str)
                if mv not in board.legal_moves:
                    return
            except ValueError:
                return
            tk = board._transposition_key()
            if board.is_zeroing(mv):
                seen.clear()
            else:
                seen[tk] = seen.get(tk, 0) + 1
            board.push(mv)
            seq.append(cu.board_codes(board))

        self.board = board
        self.codes_hist = seq
        self.seen_keys = seen
        self.pending_bestmove = None
        self.pending_search = False

    def cmd_setoption(self, line: str):
        try:
            after = line.split("name", 1)[1].strip()
            name, _, val = after.partition("value")
            name, val = name.strip().lower(), val.strip()
        except IndexError:
            return
        try:
            if name == "elo":
                self.self_elo = self.oppo_elo = float(val)
            elif name == "selfelo":
                self.self_elo = float(val)
            elif name == "oppoelo":
                self.oppo_elo = float(val)
            elif name == "title":
                self.self_title = self.oppo_title = cu.parse_title_tier(val)
            elif name == "selftitle":
                self.self_title = cu.parse_title_tier(val)
            elif name == "oppotitle":
                self.oppo_title = cu.parse_title_tier(val)
            elif name == "format":
                self.game_format = {"blitz": 0, "rapid": 1, "otb": 2}.get(
                    val.lower(), self.game_format
                )
            elif name == "temperature":
                self.temperature = float(val)
            elif name == "minp":
                self.min_p = float(val)
            elif name == "maxwdldrop":
                self.max_wdl_drop = float(val)
            elif name == "multipv":
                self.multipv = max(1, min(20, int(val)))
        except ValueError:
            return

    def run(self):
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            head = line.split()[0]
            if line == "uci":
                print("id name ChessForge-Master")
                print("id author ChessForge")
                print(
                    f"option name Elo type spin default {int(self.self_elo)} min 400 max 3400"
                )
                print(
                    f"option name SelfElo type spin default {int(self.self_elo)} min 400 max 3400"
                )
                print(
                    f"option name OppoElo type spin default {int(self.oppo_elo)} min 400 max 3400"
                )
                print("option name Title type string default GM")
                print("option name Format type string default otb")
                print(f"option name Temperature type string default {self.temperature}")
                print(f"option name MinP type string default {self.min_p}")
                print(f"option name MaxWDLDrop type string default {self.max_wdl_drop}")
                print(
                    f"option name MultiPV type spin default {self.multipv} min 1 max 20"
                )
                print("uciok", flush=True)
            elif line == "isready":
                print("readyok", flush=True)
            elif line == "ucinewgame":
                self._reset_game()
            elif head == "setoption":
                self.cmd_setoption(line)
            elif head == "position":
                self.cmd_position(line)
            elif head == "go":
                mv, top = self.search(line)
                for rank, item in enumerate(top, start=1):
                    w, d, l = item["wdl"]
                    cp = w - l
                    print(
                        f"info depth 1 multipv {rank} score cp {cp} wdl {w} {d} {l} "
                        f"pv {item['move'].uci()}",
                        flush=True,
                    )
                if "infinite" in line.split():
                    self.pending_bestmove = mv
                    self.pending_search = True
                else:
                    print(f"bestmove {mv.uci() if mv else '0000'}", flush=True)
            elif line == "stop":
                if self.pending_search:
                    mv = self.pending_bestmove
                    print(f"bestmove {mv.uci() if mv else '0000'}", flush=True)
                    self.pending_bestmove = None
                    self.pending_search = False
            elif line == "quit":
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to training YAML config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument(
        "--released", default="maia3-5m", help="Fallback if --config omitted"
    )
    ap.add_argument("--elo", type=float, default=2550.0)
    ap.add_argument("--title", default="GM", help="GM|IM|FM|CM|NM|NONE")
    ap.add_argument("--format", default="otb", choices=["blitz", "rapid", "otb"])
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--min_p", type=float, default=0.12)
    ap.add_argument("--max_wdl_drop", type=float, default=0.08)
    ap.add_argument(
        "--effort_adaptive_temp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--multipv", type=int, default=5)
    ap.add_argument("--base_time", type=float, default=900.0)
    ap.add_argument("--inc_time", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ChessforgeUCIEngine(ap.parse_args()).run()


if __name__ == "__main__":
    main()
