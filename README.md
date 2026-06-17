# 選擇權定價（二項式 / FDM / 美式）

選擇權課程的程式作業，涵蓋二項式模型、有限差分法 (FDM) 與美式選擇權定價。

## 程式
| 檔案 | 說明 |
|------|------|
| `How_to_define_the_derivative_price_under_the_complete_binomial_model.ipynb` | 二項式模型推導衍生品定價（參考 Björk《Arbitrage Theory in Continuous Time》）|
| `American_pricing.ipynb` | 美式選擇權定價（動態規劃 / RL 求最適履約邊界）|
| `選擇權作業.ipynb` | 課程數值作業 |
| `FDM筆記 PPT/` | 自製的 FDM 講義與選擇權定價程式（`fdm_option_pricing.py`、`.tex`／PDF）|

## 執行
```bash
pip install numpy scipy matplotlib
```
`American_pricing.ipynb` 另需課程提供的 `rl` 套件（非 PyPI 標準套件，需另行取得）。

> 本 repo 僅含作者自己撰寫的程式與筆記；課程講義與第三方教材因版權未收錄。
