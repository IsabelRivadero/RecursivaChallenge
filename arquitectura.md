desafio_lead_ds/
├── .dockerignore
├── .gitignore
├── Dockerfile       # Contenedor para ejecución en AWS / Local
├── README.md                   # Documentación técnica de arquitectura y ejecución
├── pyproject.toml              # Definición de proyecto y metadatos de empaquetado
├── requirements.txt            # Dependencias fijadas para el entorno de producción
├── config/
│   └── config.yaml             # Parámetros del modelo, rutas y reglas de negocio
├── data/                       # Datasets
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── desafio_lead_ds.ipynb   # Notebook con análisis exploratorio y prototipado
└── src/                        # Paquete modular Python
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── loader.py           # Ingesta y validación de esquemas (Pandas/Pydantic)
    ├── features/
    │   ├── __init__.py
    │   └── builder.py          # Ingeniería de variables (Lags, Rolling features, Calendario)
    ├── metrics/
    │   ├── __init__.py
    │   └── evaluator.py        # Cálculo estandarizado de WAPE, Bias, OTIF
    ├── models/
    │   ├── __init__.py
    │   ├── train.py            # Rolling-origin backtest y entrenamiento de modelos
    │   └── predict.py          # Pipeline de inferencia para producción
    ├── resilience/
    │   ├── __init__.py
    │   └── gate.py             # Control de resguardo de historia mínima (Cold Start)
    └── main.py                 # Script orquestador principal (Punto de entrada CLI/Batch)