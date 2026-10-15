"""Exercise ArrayRecord reading and the callable Grain feature transform."""

from pathlib import Path

import numpy as np

from chessforge.config import DataConfig
from chessforge.data import pipeline, records
from chessforge.data.pgn import ShardedWriter


def test_array_record_train_and_eval_batches(tmp_path: Path):
    writer = ShardedWriter(str(tmp_path), "positions", records_per_shard=2)
    try:
        for move in range(3):
            writer.write(
                records.encode_position(
                    boards=np.zeros((2, 64), np.int8),
                    n_hist=1,
                    stm=0,
                    move=move,
                    self_elo=1500,
                    oppo_elo=1600,
                    value=1,
                    ply=2,
                    legal=[move],
                )
            )
    finally:
        writer.close()

    pattern = str(tmp_path / "*.array_record")
    assert len(list(tmp_path.glob("*.array_record"))) == 2
    batches = list(pipeline.eval_batches(pattern, history=2, global_batch_size=2))
    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0]["move"], [0, 1])
    np.testing.assert_array_equal(batches[1]["move"], [2, 0])
    np.testing.assert_array_equal(batches[1]["weight"], [1, 0])
    np.testing.assert_array_equal(batches[0]["legal"].sum(axis=1), [1, 1])
    assert batches[1]["legal"][0, 2]
    assert batches[1]["legal"][1].all()

    config = DataConfig(
        train_glob=pattern,
        global_batch_size=2,
        history_mask_prob=0.0,
        num_workers=0,
        read_threads=1,
        prefetch_buffer_size=2,
    )
    iterator, count = pipeline.make_train_iterator(config, history=2)
    assert count == 3
    batch = next(iterator)
    assert batch["board"].shape == (2, 2, 64)
    np.testing.assert_array_equal(batch["weight"], [1, 1])
    np.testing.assert_array_equal(batch["legal"].sum(axis=1), [1, 1])
