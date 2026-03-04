# 🫀 ECV Pediátrico — Dashboard de Riesgo Cardiovascular

Dashboard profesional para predicción de riesgo cardiovascular en población pediátrica, desarrollado con **Dash (Plotly)** y **scikit-learn**.

## 📋 Descripción

Sistema de apoyo clínico que integra un modelo de Machine Learning (Random Forest) para:

- **Predicción de riesgo** cardiovascular en pacientes pediátricos
- **Exploración interactiva** del dataset
- **Evaluación del modelo** con métricas y curvas
- **Simulación clínica** con semáforo de riesgo
- **Explicabilidad** con SHAP values
- **Monitoreo y auditoría** de predicciones

## 🏗️ Estructura del Proyecto

```
ecv_pediatric_dashboard/
├── app.py                          # Punto de entrada principal
├── gunicorn_config.py              # Configuración de Gunicorn
├── requirements.txt                # Dependencias
├── .env.example                    # Variables de entorno
├── assets/
│   └── styles.css                  # Design system CSS
├── data/
│   ├── generate_sample_data.py     # Generador de datos y modelo
│   ├── dataset_ecv_pediatrico.csv  # Dataset (generado)
│   └── audit_log.csv               # Log de auditoría
├── models/
│   ├── modelo_final_GradientBoosting.joblib  # Modelo entrenado
│   └── model_metrics.json               # Métricas del modelo
├── layout/
│   ├── __init__.py
│   ├── sidebar.py                  # Navegación lateral
│   ├── home.py                     # Página 1: Resumen Ejecutivo
│   ├── eda.py                      # Página 2: Exploración de Datos
│   ├── model_evaluation.py         # Página 3: Evaluación del Modelo
│   ├── clinical_simulator.py       # Página 4: Simulador Clínico
│   ├── explainability.py           # Página 5: Explicabilidad (SHAP)
│   └── monitoring.py               # Página 6: Monitoreo y Auditoría
├── callbacks/
│   ├── __init__.py
│   ├── navigation.py               # Routing entre páginas
│   ├── eda_callbacks.py            # Callbacks de EDA
│   ├── model_eval_callbacks.py     # Callbacks de evaluación
│   ├── simulator_callbacks.py      # Callbacks del simulador
│   ├── explainability_callbacks.py # Callbacks de SHAP
│   └── monitoring_callbacks.py     # Callbacks de monitoreo
└── utils/
    ├── __init__.py
    ├── model_loader.py              # Carga de modelo (singleton)
    ├── data_loader.py               # Carga de datos
    ├── metrics.py                   # Cálculo de métricas
    └── audit_logger.py              # Sistema de auditoría
```

## 🚀 Instalación Rápida

### 1. Clonar o copiar el proyecto

```bash
cd ecv_pediatric_dashboard
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Generar datos y modelo de ejemplo

```bash
python data/generate_sample_data.py
```

### 5. Ejecutar el dashboard

```bash
python app.py
```

Abrir en el navegador: **http://localhost:8050**

## 🖥️ Páginas del Dashboard

| # | Página | Descripción |
|---|--------|-------------|
| 1 | **Resumen Ejecutivo** | KPIs del modelo, prevalencia, feature importances |
| 2 | **Exploración de Datos** | Filtros, histogramas, boxplots, correlación, tabla |
| 3 | **Evaluación del Modelo** | ROC, PR, comparación, threshold interactivo |
| 4 | **Simulador Clínico** | Formulario de entrada + semáforo de riesgo |
| 5 | **Explicabilidad** | SHAP global y por paciente (waterfall) |
| 6 | **Monitoreo** | Log de predicciones, métricas por subgrupo, alertas |

## 📊 Variables del Modelo

| Variable | Tipo | Rango |
|----------|------|-------|
| `sexo` | Binario | 0 (F) / 1 (M) |
| `edad` | Continua | 2-17 años |
| `colesterol_total` | Continua | 100-300 mg/dL |
| `presion_sistolica` | Entera | 70-160 mmHg |
| `presion_diastolica` | Entera | 40-100 mmHg |
| `frecuencia_cardiaca` | Entera | 50-140 lpm |

**Target:** `riesgo_alto` (0/1)

---

## 🔧 Uso con Modelo Propio

Si ya tiene un modelo entrenado (`modelo_final_GradientBoosting.joblib`):

1. Coloque el archivo en `models/`
2. Genere el archivo de métricas `model_metrics.json` con la estructura esperada (ver ejemplo en `data/generate_sample_data.py`)
3. Coloque su dataset como `data/dataset_ecv_pediatrico.csv`
4. Ejecute `python app.py`

---

## 📄 Licencia

Proyecto académico — Uso educativo.
