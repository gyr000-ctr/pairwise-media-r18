# codec

## 题面（可直接粘贴）

子目录：codec。ADPCM 编解码。约束：编码-解码往返与原始信号 SNR ≥40 dB；码率固定为每样本 4 bit；首样本原样保留；步长自适应按标准表更新（≤1e-9）；非法输入可区分报错；100 万样本编解码不超过 600 ms；并发安全。验收：cd codec && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd codec && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
