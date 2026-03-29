# Dataset Factory

Generador profesional de datasets OHLCV con una sola arquitectura viva, saneamiento general obligatorio, saneamiento Qlib opcional/controlado, validacion interna, validacion externa pluggable y artefactos ordenados bajo `workspace/`.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Estructura operativa

Todo lo generado en ejecucion se escribe bajo `workspace/`:

- `workspace/runs/<run_id>/`
- `workspace/exports/<run_id>/`
- `workspace/manifests/<run_id>/`
- `workspace/reports/<run_id>/`
- `workspace/temp/<run_id>/`
- `workspace/audits/`

`run_id` usa timestamp UTC con microsegundos y sufijo aleatorio corto para evitar colisiones.

## Abrir Streamlit

```powershell
.\scripts\launch_streamlit.bat
```

O directamente:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

La UI permite:

- ticker unico o varios tickers
- rango exacto o anos moviles
- intervalo
- listing preference
- modo DQ
- extras por checkbox
- casilla `Saneamiento Qlib`
- boton `Generar dataset`

Si el preset es `qlib`, la casilla `Saneamiento Qlib` se activa y se bloquea automaticamente.

## CLI single

```powershell
.\.venv\Scripts\python.exe .\export_ohlcv_csv.py --ticker MSFT --start 2018-01-01 --end 2025-01-01 --outdir .\workspace --mode base --dq-mode report
```

## CLI batch

```powershell
.\.venv\Scripts\python.exe .\export_ohlcv_csv.py --tickers "MSFT,AAPL,NVDA" --start 2018-01-01 --end 2025-01-01 --outdir .\workspace --mode extended --extras adj_close,dividends,stock_splits
```

```powershell
.\.venv\Scripts\python.exe .\export_ohlcv_csv.py --tickers-file .\tests\fixtures\universe.txt --years 5 --outdir .\workspace --mode base
```

## Salida Qlib-ready

Preset Qlib directo:

```powershell
.\.venv\Scripts\python.exe .\export_ohlcv_csv.py --tickers "MSFT,AAPL" --start 2018-01-01 --end 2025-01-01 --outdir .\workspace --mode qlib
```

Artefacto Qlib en paralelo sobre una salida general:

```powershell
.\.venv\Scripts\python.exe .\export_ohlcv_csv.py --ticker MSFT --years 5 --outdir .\workspace --mode extended --extras adj_close --qlib-sanitization
```

Cuando el flujo Qlib esta activo, el sistema:

- ejecuta saneamiento general
- calcula `factor` con politica trazable basada en `stock_splits`
- emite un CSV compatible con Qlib
- genera sidecar tecnico con la politica de factor y razones de compatibilidad

La salida Qlib-ready queda preparada para los dos pasos finales externos:

1. `dump_bin.py`
2. `check_data_health.py`

## Limpieza e higiene

Limpiar `workspace/`:

```powershell
.\.venv\Scripts\python.exe .\scripts\clean_workspace.py
```

Escanear fugas personales, rutas locales y secretos:

```powershell
.\.venv\Scripts\python.exe .\scripts\scrub_personal_data.py
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Con cobertura:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --cov=dataset_core --cov-report=term-missing --cov-report=xml:workspace\reports\coverage.xml
```

## Docs

- `docs/architecture/README.md`
- `docs/examples/README.md`
- `docs/qlib/README.md`
