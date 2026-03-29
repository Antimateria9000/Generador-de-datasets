# Arquitectura Viva

El proyecto mantiene una sola linea de ejecucion activa:

1. `export_ohlcv_csv.py` construye `DatasetRequest` y delega en `dataset_core.batch_orchestrator`.
2. `dataset_core.batch_orchestrator` crea un `run_id` robusto, prepara directorios de `workspace/` y ejecuta el lote ticker a ticker.
3. `dataset_core.export_service` coordina adquisicion, saneamiento, validacion, serializacion y manifiestos.

Responsabilidades por paquete:

- `providers/`
  - `yfinance_provider.py`: descarga OHLCV y eventos corporativos.
  - `market_context.py`: resolucion de contexto de mercado y listing preference.
- `dataset_core/`
  - `acquisition.py`: servicio de adquisicion desacoplado del provider concreto.
  - `sanitization_general.py`: saneamiento canonico obligatorio para toda salida.
  - `sanitization_qlib.py`: adaptacion especifica para Qlib sobre el dataset canonico ya saneado.
  - `factor_policy.py`: politica trazable de `factor` y de ajuste OHLCV/volumen.
  - `schema_builder.py`: presets visibles de salida y orden final de columnas.
  - `validation_internal.py`: data quality interna reutilizable.
  - `validation_external.py` + `reference_adapters.py`: validacion externa pluggable.
  - `serialization.py`: escritura ordenada, sidecars y hashes.
  - `manifest_service.py` + `result_models.py`: contratos de resultado y manifiestos.
- `app/`
  - `streamlit_app.py`: UX unica para ejecucion interactiva.
- `scripts/`
  - lanzadores y utilidades de limpieza/scrub del repo.

Politica de artefactos:

- Todo lo generado en ejecucion vive bajo `workspace/`.
- La raiz contiene solo codigo, configuracion, documentacion y lanzadores minimos.
- No existe codigo legacy operativo ni carpetas historicas activas.
