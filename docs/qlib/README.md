# Flujo Qlib

El flujo Qlib no reemplaza al dataset canonico. Primero se ejecuta el saneamiento general y despues, solo cuando corresponde, la adaptacion Qlib.

Activacion:

- Preset `qlib`: activa y bloquea `Saneamiento Qlib`.
- Presets `base` o `extended` + casilla `Saneamiento Qlib`: mantienen la salida general y emiten un artefacto Qlib paralelo.

Contrato Qlib emitido:

- columnas minimas: `date, open, high, low, close, volume, factor`
- un CSV por simbolo con naming `TICKER.csv`
- fechas ordenadas y sin duplicados
- sidecar tecnico `*.qlib.json` con politica de factor, warnings y compatibilidad

Politica de factor:

- el provider se fuerza a `auto_adjust=False` cuando el flujo requiere `factor`
- el provider se fuerza a `actions=True` para recuperar splits
- `factor` se reconstruye desde `stock_splits`
- la salida Qlib ajusta `open/high/low/close` por `factor`
- `volume` se ajusta con la inversa del `factor`

Artefactos:

- dataset canonico: `workspace/exports/<run_id>/canonical/`
- dataset general visible: `workspace/exports/<run_id>/csv/`
- dataset Qlib-ready: `workspace/exports/<run_id>/qlib/`
- sidecars y manifiestos: `workspace/manifests/<run_id>/` y `workspace/reports/<run_id>/`

Pasos finales esperados fuera del proyecto:

1. `dump_bin.py`
2. `check_data_health.py`

El proyecto deja el CSV y el sidecar Qlib listos para esos dos pasos finales. Si el entorno donde se ejecutan esas herramientas requiere una version concreta de Python o de Qlib, esa compatibilidad se resuelve en ese entorno final, no alterando la arquitectura del generador.
