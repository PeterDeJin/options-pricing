# 選擇權理論與應用

選擇權課程的程式、講義與作業：涵蓋定價（Black-Scholes、二項式/樹模型、有限差分法 FDM）、Greeks、交易策略、隨機波動度與美式選擇權。

## 程式
| 檔案 | 說明 |
|------|------|
| `How_to_define_the_derivative_price_under_the_complete_binomial_model.ipynb` | 二項式模型推導衍生品定價（參考 Björk《Arbitrage Theory in Continuous Time》）|
| `American_pricing.ipynb` | 美式選擇權定價（動態規劃 / RL 求最適履約邊界）|
| `選擇權作業.ipynb` | 課程數值作業 |
| `FDM筆記 PPT/fdm_option_pricing.py` | 有限差分法 (FDM) 選擇權定價 |

## 講義與參考（PDF / PPT）
Black-Scholes 理論與應用、FDM 講義與手寫筆記、Greeks 與策略、Put-Call Parity、隨機波動度、樹模型、期中參考、TA 補充等；`FDM筆記 PPT/` 為自製的 FDM 講義（含 `.tex` 原始檔）。

## 執行
```bash
pip install numpy scipy matplotlib
```

> 注意：`American_pricing.ipynb` 使用了 `rl` 套件（reinforcement learning 課程用），執行前需確認該相依套件來源。

---
> 註：本 repo 僅含作者自己的程式與「FDM筆記 PPT」筆記；課程講義與第三方教材（PDF/PPT）因版權未納入。
