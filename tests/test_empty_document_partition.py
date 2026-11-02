"""An empty document partition must fail instead of cycling forever."""
import multiprocessing
import os

import pyarrow as pa
import pyarrow.parquet as pq


def _read_one(connection, paths, rank, world_size, resume):
    try:
        os.environ["NANOCHAT_BASE_DIR"] = os.path.dirname(paths[0])
        from nanochat import dataloader
        dataloader.list_parquet_files = lambda **kwargs: paths
        dataloader.get_dist_info = lambda: (world_size > 1, rank, rank, world_size)
        connection.send(("ready",))
        try:
            result = next(dataloader._document_batches("train", resume, 2))
        except Exception as error:
            connection.send(("error", type(error).__name__, str(error)))
        else:
            connection.send(("batch", result))
    except Exception as error:
        connection.send(("startup_error", type(error).__name__, str(error)))
    finally:
        connection.close()


def probe(paths, rank=0, world_size=1, resume=None):
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    worker = context.Process(target=_read_one, args=(send, [str(path) for path in paths], rank, world_size, resume), daemon=True)
    worker.start()
    send.close()
    try:
        assert receive.poll(20), "worker import did not complete"
        assert receive.recv() == ("ready",)
        return receive.recv() if receive.poll(2) else ("timeout",)
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(10)
        if worker.is_alive():
            worker.kill()
            worker.join(5)
        receive.close()


def parquet(path, texts):
    pq.write_table(pa.table({"text": pa.array(texts, type=pa.string())}), path, row_group_size=1)
    return path


def test_rank_without_any_row_group_fails_instead_of_spinning(tmp_path):
    paths = [parquet(tmp_path / "train.parquet", ["first"]), parquet(tmp_path / "val.parquet", ["validation"])]
    result = probe(paths, rank=1, world_size=2)
    assert result[0:2] == ("error", "ValueError"), result
    assert "rank 1" in result[2]


def test_validation_only_cache_has_no_training_partition(tmp_path):
    result = probe([parquet(tmp_path / "val.parquet", ["validation"])])
    assert result[0:2] == ("error", "ValueError"), result
    assert "train" in result[2]


def test_empty_row_group_fails_instead_of_spinning(tmp_path):
    paths = [parquet(tmp_path / "train.parquet", []), parquet(tmp_path / "val.parquet", ["validation"])]
    result = probe(paths)
    assert result[0:2] == ("error", "ValueError"), result


def test_valid_rank_still_reads_its_assigned_row_group(tmp_path):
    paths = [parquet(tmp_path / "train.parquet", ["first", "second"]), parquet(tmp_path / "val.parquet", ["validation"])]
    assert probe(paths, rank=1, world_size=2) == ("batch", (["second"], (0, 1, 1)))


def test_resume_after_last_row_group_wraps_to_next_epoch(tmp_path):
    paths = [parquet(tmp_path / "train.parquet", ["first"]), parquet(tmp_path / "val.parquet", ["validation"])]
    resume = {"pq_idx": 0, "rg_idx": 0, "epoch": 7}
    assert probe(paths, resume=resume) == ("batch", (["first"], (0, 0, 8)))
