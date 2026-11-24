"""texatlas 核心 API（初始未实现）。"""


class TexAtlas:
    def __init__(self, size, pad):
        """构造"""
        raise NotImplementedError

    def add(self, name, w, h):
        """加入"""
        raise NotImplementedError

    def uv(self, name, mip=0):
        """查 UV"""
        raise NotImplementedError
