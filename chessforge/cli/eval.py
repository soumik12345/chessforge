"""Evaluate either a released Maia-3 checkpoint (validation of this codebase) or our own run.

python -m chessforge.cli.eval --released maia3-5m --test_glob '/data/allie_test/*.array_record'
python -m chessforge.cli.eval --config configs/maia3_5m.yaml
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--released", help="maia3-3m|maia3-5m|maia3-23m|maia3-79m")
    ap.add_argument("--compute_dtype", default="float32")
    ap.add_argument("--test_glob")
    ap.add_argument("--step", type=int)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--out")
    a = ap.parse_args()

    import jax

    try:
        jax.distributed.initialize()
    except Exception:  # noqa: BLE001
        pass
    from ..config import load_train_config
    from ..train.trainer import Trainer

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
    result = trainer.evaluate(params, trainer.cfg.data.test_glob)
    if jax.process_index() == 0:
        print(json.dumps(result["overall"], indent=2))
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
