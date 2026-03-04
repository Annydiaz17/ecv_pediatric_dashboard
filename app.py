"""
ECV Pediátrico Dashboard — Aplicación Principal
=================================================
Dashboard profesional para predicción de riesgo cardiovascular pediátrico.
Reestructurado en 4 pestañas según la metodología SEMMA.

Ejecución en desarrollo:
    python app.py

Ejecución en producción:
    gunicorn -c gunicorn_config.py app:server
"""
import os
from dash import Dash, html, dcc
import dash

# ─── Configuración ───────────────────────────────────────────
DEBUG = os.getenv("DASH_DEBUG", "True").lower() == "true"
HOST = os.getenv("DASH_HOST", "127.0.0.1")
PORT = int(os.getenv("DASH_PORT", 8050))

# ─── Inicialización de la App ────────────────────────────────
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    update_title="Cargando...",
    title="ECV Pediátrico · Dashboard de Riesgo Cardiovascular",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        {"name": "description", "content": "Sistema de apoyo clínico para predicción de riesgo cardiovascular pediátrico basado en Machine Learning — Metodología SEMMA"},
        {"charset": "UTF-8"},
    ],
)

# Servidor Flask para Gunicorn
server = app.server

# ─── Layout Principal ────────────────────────────────────────
from layout.sidebar import create_sidebar

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),

    # Sidebar
    create_sidebar(),

    # Contenido principal
    html.Div(
        id="page-content",
        className="main-content",
    ),
])

# ─── Importar TODOS los callbacks ────────────────────────────
# Es necesario importarlos para que se registren con la app
import callbacks.navigation
import callbacks.eda_callbacks
import callbacks.segmentation_callbacks
import callbacks.model_eval_callbacks
import callbacks.xai_simulator_callbacks


# ─── Punto de entrada ────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ECV Pediatrico Dashboard — SEMMA")
    print("  Sistema de Apoyo Clinico - Riesgo Cardiovascular")
    print("=" * 60)
    print(f"  URL: http://{HOST}:{PORT}")
    print(f"  Debug: {DEBUG}")
    print("  Pestañas: Resumen/EDA | Segmentacion | Evaluacion | XAI+Simulador")
    print("=" * 60)
    app.run(debug=DEBUG, host=HOST, port=PORT)
