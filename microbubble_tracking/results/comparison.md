## 各组运行

| 运行 | 模式 | 背景来源 | 平均 ms/帧 | P95 ms/帧 | 确认编号数 | 融合 | 分裂 |
|---|---|---|---|---|---|---|---|
| A_offline_noref | offline | self | 13.76 | 15.62 | 102 | 22 | 13 |
| B_offline_noref | offline | self | 14.6 | 17.25 | 65 | 4 | 5 |
| A_offline | offline | self+ref | 14.58 | 15.97 | 118 | 23 | 13 |
| B_offline | offline | self+ref | 15.22 | 17.25 | 49 | 7 | 6 |
| A_online | online | ref | 14.98 | 16.51 | 116 | 24 | 15 |
| B_online | online | ref | 14.89 | 16.81 | 48 | 7 | 7 |
| B_online_firstframe | online | first-frame | 14.76 | 16.27 | 61 | 16 | 16 |

## 与基准（离线 + 参考背景）的一致性

| 被测 | 基准 | 小目标检出率 | 小目标准确率 | 团检出率 | 团准确率 | 团总面积/帧（基准 → 被测） |
|---|---|---|---|---|---|---|
| A_online | A_offline | 98.6% | 99.4% | 100.0% | 100.0% | 5038 → 5073 |
| A_offline_noref | A_offline | 97.7% | 99.5% | 99.9% | 100.0% | 5038 → 4920 |
| B_online | B_offline | 99.3% | 99.9% | 100.0% | 100.0% | 4659 → 4657 |
| B_offline_noref | B_offline | 85.8% | 100.0% | 99.6% | 100.0% | 4659 → 4449 |
| B_online_firstframe | B_offline | 68.2% | 85.8% | 59.3% | 47.5% | 4659 → 8353 |
