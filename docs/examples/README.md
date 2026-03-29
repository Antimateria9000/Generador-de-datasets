# Ejemplos

CLI single:

```powershell
.\.venv\Scripts\python.exe .\export_ohlcv_csv.py --ticker MSFT --start 2018-01-01 --end 2025-01-01 --outdir .\workspace --mode base --dq-mode report
```

CLI batch:

```powershell
.\.venv\Scripts\python.exe .\export_ohlcv_csv.py --tickers "MSFT,AAPL,NVDA" --start 2018-01-01 --end 2025-01-01 --outdir .\workspace --mode extended --extras adj_close,dividends,stock_splits
```

CLI con fichero de tickers:

```powershell
.\.venv\Scripts\python.exe .\export_ohlcv_csv.py --tickers-file .\tests\fixtures\universe.txt --years 5 --outdir .\workspace --mode base
```

CLI Qlib-ready:

```powershell
.\.venv\Scripts\python.exe .\export_ohlcv_csv.py --ticker MSFT --start 2018-01-01 --end 2025-01-01 --outdir .\workspace --mode qlib --dq-mode off
```

Streamlit:

```powershell
.\scripts\launch_streamlit.bat
```
