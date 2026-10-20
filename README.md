# Guía de Instalación, Ejecución, Pruebas y Reproducción de Resultados

## 1. Requisitos previos

- Python 3.11 o superior.
- Docker Engine (opcional, para ejecutar el proyecto en un contenedor).
- Los siguientes archivos de entrada deben estar disponibles en el directorio data/raw/:

  - pedidos_historicos_genericos.csv
  - pronostico_demanda_generico.csv
  - capacidad_logistica_generica.csv
  - plan_entrega_clientes_generico.csv


## 2. Instalación y configuración del entorno local

Se recomienda utilizar un entorno virtual para aislar las dependencias del proyecto.

Crear y activar el entorno virtual:

    python3 -m venv venv
    source venv/bin/activate

Actualizar pip e instalar las dependencias:

    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .


## 3. Ejecución del pipeline principal

Desde la raíz del proyecto, ejecutar:

    python src/main.py

El pipeline procesa los datos de entrada, realiza la evaluación mediante validación temporal con origen rodante (rolling-origin backtesting), entrena los modelos y genera el archivo de salida:

    data/processed/pronostico_corregido.csv

Este archivo contiene las predicciones originales, las unidades reales, las variables utilizadas por el modelo y las unidades pronosticadas después de la corrección.


## 4. Resultados y reproducción de métricas

La ejecución del pipeline produjo los siguientes resultados globales:

| Métrica | Modelo base | Modelo ML corregido |
|---------|-------------|---------------------|
| WAPE    | 20.78%      | 10.83%              |
| Bias    | +6.62%      | +1.37%              |

El modelo de Machine Learning utiliza HistGradientBoostingRegressor.

### Interpretación de los resultados

- WAPE: disminuyó de 20.78% a 10.83%, lo que representa una reducción relativa del error aproximada del 47.9%.
- Bias: disminuyó de +6.62% a +1.37%, reduciendo el sesgo agregado del pronóstico.

Los resultados corresponden a la ejecución registrada del pipeline. Para reproducirlos, es necesario utilizar los datos de entrada correspondientes y las mismas versiones de las dependencias.


## 5. Ejecución mediante Docker

Docker permite ejecutar el pipeline en un contenedor aislado, sin necesidad de instalar las dependencias de Python directamente en el entorno del proyecto.

### 5.1. Construcción de la imagen

Desde la raíz del repositorio, ejecutar:

    docker build -t desafio-lead-ds:latest .

### 5.2. Ejecución del contenedor

Ejecutar el pipeline y mapear los directorios de entrada y salida:

    docker run --rm \
      -v "$(pwd)/data/raw:/app/data/raw" \
      -v "$(pwd)/data/processed:/app/data/processed" \
      desafio-lead-ds:latest

Los volúmenes permiten que el contenedor lea los archivos de entrada desde data/raw/ y guarde las predicciones en el directorio local data/processed/.

### 5.3. Verificación del archivo generado

Para comprobar que el pipeline generó correctamente el archivo de salida:

    ls -lh data/processed/

También se puede verificar su existencia con:

    test -f data/processed/pronostico_corregido.csv \
      && echo "OK: archivo de predicciones generado" \
      || echo "ERROR: no se encontró el archivo"


## 6. Pruebas unitarias

Para ejecutar las pruebas automatizadas del proyecto, utilizar:

    pytest

El comando ejecuta las pruebas configuradas en el repositorio. La ejecución exitosa de Docker no reemplaza la necesidad de ejecutar y validar las pruebas automatizadas.


## 7. Estructura de los resultados

El archivo data/processed/pronostico_corregido.csv contiene información para analizar las predicciones, incluyendo:

- Centro de distribución.
- Fecha de emisión y fecha de despacho.
- Horizonte de pronóstico.
- Unidades pronosticadas originalmente.
- Unidades reales.
- Variables temporales.
- Variables históricas y estadísticas móviles.
- Unidades pronosticadas corregidas por el modelo.

Esta información permite comparar el pronóstico original con el corregido y realizar análisis posteriores por centro de distribución y horizonte temporal.


## 8. Consideraciones

- Los archivos de entrada deben conservar los nombres y formatos esperados por el pipeline.
- Los resultados pueden variar si cambian los datos, las dependencias o la configuración del proyecto.
- Los valores nulos en las variables históricas pueden aparecer cuando no existe suficiente historial para calcular determinadas ventanas móviles.
- Las métricas globales no garantizan una mejora uniforme en todos los centros de distribución y horizontes; se recomienda evaluar también los resultados desagregados.
- La arquitectura de ejecución con Docker demuestra la capacidad de ejecutar el pipeline en un contenedor. Una eventual implementación en AWS requiere configurar y desplegar los servicios correspondientes.


## Pruebas automatizadas

El proyecto incluye pruebas unitarias con pytest para verificar:

- El cálculo de WAPE y Bias.
- La generación de variables temporales, lags y ventanas móviles.
- El comportamiento del pipeline de inferencia.
- El fallback al pronóstico original cuando no existe un modelo entrenado
  o no se dispone de historial suficiente para generar las variables requeridas.

Para instalar pytest:

    python -m pip install pytest

Para ejecutar las pruebas desde la raíz del proyecto:

    python -m pytest -v

Las pruebas utilizan datos sintéticos y modelos simulados en los casos
correspondientes, por lo que no requieren volver a entrenar los modelos
con los datasets completos.# RecursivaChallenge
