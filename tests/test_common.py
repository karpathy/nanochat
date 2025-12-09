
import os
import pytest
from unittest.mock import patch, MagicMock
from nanochat.common import (
    autodetect_device_type,
    compute_init,
    download_file_with_lock,
    is_ddp
)

@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_autodetect_device_type_cuda(mock_mps, mock_cuda):
    mock_cuda.return_value = True
    # mock device count > 0
    with patch("torch.cuda.device_count", return_value=1):
        assert autodetect_device_type() == "cuda"

@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_autodetect_device_type_mps(mock_mps, mock_cuda):
    mock_cuda.return_value = False
    mock_mps.return_value = True
    assert autodetect_device_type() == "mps"

@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_autodetect_device_type_cpu(mock_mps, mock_cuda):
    mock_cuda.return_value = False
    mock_mps.return_value = False
    assert autodetect_device_type() == "cpu"

def test_is_ddp_false():
    with patch.dict(os.environ, {}, clear=True):
        assert is_ddp() is False

def test_is_ddp_true():
    env = {
        'RANK': '0',
        'WORLD_SIZE': '2',
        'LOCAL_RANK': '0',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '1234'
    }
    with patch.dict(os.environ, env, clear=True):
        assert is_ddp() is True

@patch("urllib.request.urlopen")
@patch("nanochat.common.get_base_dir")
def test_download_file_with_lock(mock_get_base_dir, mock_urlopen, tmp_path):
    mock_get_base_dir.return_value = str(tmp_path)

    mock_response = MagicMock()
    mock_response.read.return_value = b"content"
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    filename = "test.txt"
    path = download_file_with_lock("http://example.com", filename)

    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == b"content"

@patch("torch.distributed.barrier")
@patch("torch.distributed.init_process_group")
@patch("torch.cuda.is_available")
def test_compute_init_cpu(mock_cuda, mock_init_pg, mock_barrier):
    mock_cuda.return_value = False

    # Non-DDP
    ddp, rank, local_rank, world_size, device = compute_init(device_type="cpu")
    assert ddp is False
    assert device.type == "cpu"

    # DDP
    env = {
        'RANK': '0',
        'WORLD_SIZE': '2',
        'LOCAL_RANK': '0',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '1234'
    }
    with patch.dict(os.environ, env, clear=True):
        ddp, rank, local_rank, world_size, device = compute_init(device_type="cpu")
        assert ddp is True
        assert device.type == "cpu"
        mock_init_pg.assert_called()
