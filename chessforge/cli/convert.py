"""Convert a released Maia-3 .pt to a serialized Flax msgpack."""

import argparse

from flax import nnx, serialization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ..models.convert_torch import load_released

    _, state = load_released(a.released)
    pure = nnx.to_pure_dict(state) if hasattr(nnx, "to_pure_dict") else state
    with open(a.out, "wb") as fh:
        fh.write(serialization.msgpack_serialize(pure))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
