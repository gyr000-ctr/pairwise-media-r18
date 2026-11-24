# decimate

## 题面（可直接粘贴）

子目录：decimate。二次误差网格简化。约束：简化后面数不超过目标；Hausdorff 距离不超过阈值；无退化三角形；包围盒不变（≤1e-9）；1 万面不超过 800 ms；非法目标可区分报错；并发安全。验收：cd decimate && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd decimate && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
