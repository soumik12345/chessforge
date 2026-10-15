import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="overrides, e.g. optim.lr=1e-4 data.global_batch_size=4096",
    )
    a = ap.parse_args()
    import jax

    try:
        jax.distributed.initialize()
    except Exception as e:  # noqa: BLE001  single-host / already initialized
        print(f"[info] jax.distributed.initialize skipped: {e}")
    from absl import flags

    from chessforge.config import load_train_config
    from chessforge.train.trainer import Trainer

    # This is a known issue: argparse vs abseil when working with grain
    # Without this, training fails
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["chessforge-train"])

    Trainer(load_train_config(a.config, a.set)).fit()


if __name__ == "__main__":
    main()
