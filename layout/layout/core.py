"""layout 核心 API（初始未实现）。"""


class TextLayout:
    def __init__(self, tracking=0.0):
        """构造"""
        raise NotImplementedError

    def layout(self, glyphs, max_width):
        """排版"""
        raise NotImplementedError
