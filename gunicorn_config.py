"""
Configuración de Gunicorn para producción.
Uso: gunicorn -c gunicorn_config.py app:server
"""
import multiprocessing
import os

# Bind
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8050")

# Workers: regla general = 2 * num_cores + 1
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))

# Worker class
worker_class = "sync"

# Timeout (segundos) - más alto para SHAP computations
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))

# Keep-alive
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload app para compartir modelo en memoria entre workers
preload_app = True

# Graceful restart
graceful_timeout = 30

# Max requests per worker (previene memory leaks)
max_requests = 1000
max_requests_jitter = 50
