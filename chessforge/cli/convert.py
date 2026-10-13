"""Convert a released Maia-3 .pt to a Flax msgpack (e.g. for warm starts / serving)."""
import argparse

from flax import serialization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ..models.convert_torch import load_released
    _, params = load_released(a.released)
    with open(a.out, "wb") as fh:
        fh.write(serialization.msgpack_serialize(params))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()