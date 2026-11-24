"""normalmap 核心 API（初始未实现）。"""


class NormalMap:
    def from_height(self, height, strength=1.0):
        """生成法线"""
        raise NotImplementedError

    def decode(self, rgb):
        """解码法线"""
        raise NotImplementedError
