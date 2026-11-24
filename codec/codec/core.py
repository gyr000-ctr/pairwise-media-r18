"""codec 核心 API（初始未实现）。"""


class Adpcm:
    def encode(self, samples):
        """编码为字节"""
        raise NotImplementedError

    def decode(self, data, n):
        """解码"""
        raise NotImplementedError
