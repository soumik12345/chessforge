"""Evaluate a released Maia-3 checkpoint or a Chessforge v1/v2 checkpoint."""

import argparse
import json
import os

import numpy as np


def _make_mask_transform(
    mask_clock: bool, mask_titles: bool, mask_format: bool, mask_markov: bool
):
    if not (mask_clock or mask_titles or mask_format or mask_markov):
        return None

    def _transform(b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(b)
        if mask_clock:
            out["rich_time"] = np.zeros_like(b["rich_time"])
            out["has_clock"] = np.zeros_like(b["has_clock"])
            out["time_info"] = np.zeros_like(b["time_info"])
        if mask_titles:
            out["self_title"] = np.zeros_like(b["self_title"])
            out["oppo_title"] = np.zeros_like(b["oppo_title"])
        if mask_format:
            out["game_format"] = np.zeros_like(b["game_format"])
        if mask_markov:
            out["board_meta"] = np.zeros_like(b["board_meta"])
            out["ep_square_idx"] = np.full_like(b["ep_square_idx"], -1)
        return out

    return _transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--released", help="maia3-3m|maia3-5m|maia3-23m|maia3-79m")
    ap.add_argument("--compute_dtype", default="float32")
    ap.add_argument("--test_glob")
    ap.add_argument("--step", type=int)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--mask_clock", action="store_true")
    ap.add_argument("--mask_titles", action="store_true")
    ap.add_argument("--mask_format", action="store_true")
    ap.add_argument("--mask_markov", action="store_true")
    ap.add_argument(
        "--mask_all_context",
        action="store_true",
        help="Strict MAIA-3 info parity: zero clocks, official titles, format, and Markov meta",
    )
    ap.add_argument("--out")
    a = ap.parse_args()

    from chessforge.distributed import init_distributed

    init_distributed()

    import jax
    from absl import flags

    from ..config import load_train_config
    from ..train.trainer import Trainer

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["chessforge-eval"])

    overrides = list(a.set) + [f"data.eval_batch_size={a.batch_size}"]
    if a.test_glob:
        overrides.append(f"data.test_glob={a.test_glob}")
    if a.released:
        from ..models.convert_torch import load_released

        overrides += [
            f"model={a.released}",
            f"model_overrides={{compute_dtype: {a.compute_dtype}}}",
        ]
        trainer = Trainer(load_train_config(a.config, overrides))
        _, params = load_released(a.released, a.compute_dtype)
        params = jax.device_put(params, trainer.repl)
    else:
        trainer = Trainer(load_train_config(a.config, overrides))
        params = trainer.restore_params(a.step)

    tx = _make_mask_transform(
        a.mask_clock or a.mask_all_context,
        a.mask_titles or a.mask_all_context,
        a.mask_format or a.mask_all_context,
        a.mask_markov or a.mask_all_context,
    )
    result = trainer.evaluate(params, trainer.cfg.data.test_glob, batch_transform=tx)
    if jax.process_index() == 0:
        print(json.dumps(result["overall"], indent=2))
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as fh:
                json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()