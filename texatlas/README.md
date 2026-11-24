# texatlas

## 题面（可直接粘贴）

子目录：texatlas。纹理图集 UV 查询与出血。约束：UV 矩形与打包结果一致（≤1e-12）；带 padding 时采样不越界；mip 级别下 UV 收缩正确（避免接缝）；非法名字可区分报错；10 万次查询不超过 100 ms；并发安全。验收：cd texatlas && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd texatlas && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
