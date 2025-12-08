
import os
import shutil
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import patch, MagicMock
from nanochat.dataset import download_single_file, verify_file, DATA_DIR, index_to_filename

# Mock the DATA_DIR to a temporary directory
@pytest.fixture
def mock_data_dir(tmp_path):
    with patch("nanochat.dataset.DATA_DIR", str(tmp_path)):
        yield tmp_path

def create_valid_parquet(filepath):
    table = pa.Table.from_pydict({'a': [1, 2], 'b': ['x', 'y']})
    pq.write_table(table, filepath)
    return table

def test_verify_file_valid(mock_data_dir):
    # Create a valid parquet file
    filepath = os.path.join(mock_data_dir, "valid.parquet")
    create_valid_parquet(filepath)

    assert verify_file(filepath) is True

def test_verify_file_corrupt(mock_data_dir):
    # Create a corrupt file
    filepath = os.path.join(mock_data_dir, "corrupt.parquet")
    with open(filepath, "wb") as f:
        f.write(b"garbage")

    assert verify_file(filepath) is False

def test_download_single_file_skips_valid(mock_data_dir, capsys):
    # Setup a valid file
    filename = index_to_filename(0)
    filepath = os.path.join(mock_data_dir, filename)
    create_valid_parquet(filepath)

    result = download_single_file(0)

    assert result is True
    captured = capsys.readouterr()
    assert "already exists" in captured.out

def test_download_single_file_redownloads_corrupt(mock_data_dir, capsys):
    # Setup a corrupt file
    filename = index_to_filename(0)
    filepath = os.path.join(mock_data_dir, filename)
    with open(filepath, "wb") as f:
        f.write(b"garbage")

    # Generate actual parquet bytes
    import io
    buf = io.BytesIO()
    table = pa.Table.from_pydict({'a': [1], 'b': ['x']})
    pq.write_table(table, buf)
    parquet_bytes = buf.getvalue()

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [parquet_bytes]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = download_single_file(0)

    assert result is True
    captured = capsys.readouterr()
    assert "is corrupt, re-downloading" in captured.out
    assert "Successfully downloaded" in captured.out
    # Check that the file on disk is now valid
    assert verify_file(filepath) is True
