import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest

import texatlas


@pytest.fixture(autouse=True)
def _reset_texatlas_registry():
    """每个用例前清空共享打包注册表，保证用例间相互隔离。"""
    texatlas.reset()
    yield
    texatlas.reset()
