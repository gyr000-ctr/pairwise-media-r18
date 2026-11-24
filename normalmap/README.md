# normalmap

## 题面（可直接粘贴）

子目录：normalmap。高度图转法线图。约束：单位化（≤1e-6）；平坦处法线为 +Z；强度参数线性影响 xy 分量；边界 clamp ；编码到 [0,1] 后可逆（≤1/255）；非法强度可区分报错；100 万像素不超过 500 ms。验收：cd normalmap && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd normalmap && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
