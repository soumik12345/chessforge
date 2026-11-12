import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="overrides, e.g. optim.lr=4e-4 data.global_batch_size=2048",
    )
    a = ap.parse_args()

    from chessforge.distributed import init_distributed

    init_distributed()

    from absl import flags

    from chessforge.config import load_train_config
    from chessforge.train.trainer import Trainer

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["chessforge-train"])

    Trainer(load_train_config(a.config, a.set)).fit()


if __name__ == "__main__":
    main()
