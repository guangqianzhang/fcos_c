import ctypes
import glob
import os
import logging
import json
import re
import sys
from deploy.logging import get_logger


def search_cuda_version() -> str:
    """try cmd to get cuda version, then try `torch.cuda`

    Returns:
        str: cuda version, for example 10.2
    """

    version = None

    pattern = re.compile(r'[0-9]+\.[0-9]+')
    platform = sys.platform.lower()

    def cmd_result(txt: str):
        cmd = os.popen(txt)
        return cmd.read().rstrip().lstrip()

    if platform == 'linux' or platform == 'darwin' or platform == 'freebsd':  # noqa E501
        version = cmd_result(
            " nvcc --version | grep  release | awk '{print $5}' | awk -F , '{print $1}' "  # noqa E501
        )
        if version is None or pattern.match(version) is None:
            version = cmd_result(
                " nvidia-smi  | grep CUDA | awk '{print $9}' ")

    elif platform == 'win32' or platform == 'cygwin':
        # nvcc_release = "Cuda compilation tools, release 10.2, V10.2.89"
        nvcc_release = cmd_result(' nvcc --version | find "release" ')
        if nvcc_release is not None:
            result = pattern.findall(nvcc_release)
            if len(result) > 0:
                version = result[0]

        if version is None or pattern.match(version) is None:
            # nvidia_smi = "| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |" # noqa E501
            nvidia_smi = cmd_result(' nvidia-smi | find "CUDA Version" ')
            result = pattern.findall(nvidia_smi)
            if len(result) > 2:
                version = result[2]

    if version is None or pattern.match(version) is None:
        try:
            import torch
            version = torch.version.cuda
        except Exception:
            pass

    return version

def get_root_logger(log_file=None, log_level=logging.INFO) -> logging.Logger:
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        logging.Logger: The obtained logger
    """
    logger = get_logger(
        name='mmdeploy', log_file=log_file, log_level=log_level)

    return logger
def get_file_path(prefix, candidates) -> str:
    """Search for file in candidates.

    Args:
        prefix (str): Prefix of the paths.
        candidates (str): Candidate paths
    Returns:
        str: file path or '' if not found
    """
    for candidate in candidates:
        wildcard = os.path.abspath(os.path.join(prefix, candidate))
        paths = glob.glob(wildcard)
        if paths:
            lib_path = paths[0]
            return lib_path
    return ''
def get_ops_path() -> str:
    """Get path of the TensorRT plugin library.

    Returns:
        str: A path of the TensorRT plugin library.
    """
    candidates = [
        '/home/cqjtu/PycharmProjects/cdc/mmdeploy/lib/libmmdeploy_tensorrt_ops.so',
        '/home/cqjtu/PycharmProjects/cdc//mmdeploy/lib/mmdeploy_tensorrt_ops.dll',
        '/home/cqjtu/PycharmProjects/cdc/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so',
        '/home/cqjtu/PycharmProjects/cdc/mmdeploy/build/bin/*/mmdeploy_tensorrt_ops.dll'
    ]
    return get_file_path(os.path.dirname(__file__), candidates)

def load_tensorrt_plugin() -> bool:
    """Load TensorRT plugins library.

    Returns:
        bool: True if TensorRT plugin library is successfully loaded.
    """
    lib_path = get_ops_path()
    success = False
    logger = get_root_logger()
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        logger.info(f'Successfully loaded tensorrt plugins from {lib_path}')
        success = True
    else:
        logger.warning(f'Could not load the library of tensorrt plugins.'
                       f'Because the file does not exist: {lib_path}')
    return success

def read_json_file(jsonfile):
    with open(jsonfile) as f:
        data = json.load(f)
        return data
    

