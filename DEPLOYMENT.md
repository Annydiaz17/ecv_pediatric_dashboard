# 🚀 Guía de Despliegue en Ubuntu

## Requisitos del Servidor

- Ubuntu 20.04+ (LTS recomendado)
- Python 3.9+
- 2GB RAM mínimo (4GB recomendado para SHAP)
- Acceso SSH al servidor

---

## 1. Preparar el Servidor

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Python y dependencias del sistema
sudo apt install -y python3 python3-pip python3-venv git nginx

# Verificar versión de Python
python3 --version
```

## 2. Configurar el Proyecto

```bash
# Crear directorio del proyecto
sudo mkdir -p /opt/ecv-dashboard
sudo chown $USER:$USER /opt/ecv-dashboard

# Copiar archivos del proyecto
# (usar scp, git clone, o el método que prefiera)
cp -r ./ecv_pediatric_dashboard/* /opt/ecv-dashboard/

# Ir al directorio
cd /opt/ecv-dashboard

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# Generar datos y modelo (solo si no tiene los archivos)
python data/generate_sample_data.py
```

## 3. Configurar Variables de Entorno

```bash
# Copiar template
cp .env.example .env

# Editar configuración
nano .env
```

Configuración recomendada para producción:
```env
DASH_ENV=production
DASH_DEBUG=False
DASH_HOST=0.0.0.0
DASH_PORT=8050
GUNICORN_WORKERS=4
GUNICORN_BIND=0.0.0.0:8050
GUNICORN_TIMEOUT=120
```

## 4. Probar Ejecución Manual

```bash
cd /opt/ecv-dashboard
source venv/bin/activate

# Test rápido
python app.py
# Ctrl+C para detener

# Test con Gunicorn
gunicorn -c gunicorn_config.py app:server
# Ctrl+C para detener
```

## 5. Configurar Servicio Systemd

```bash
sudo nano /etc/systemd/system/ecv-dashboard.service
```

Contenido del archivo:
```ini
[Unit]
Description=ECV Pediatric Dashboard
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/ecv-dashboard
Environment="PATH=/opt/ecv-dashboard/venv/bin"
EnvironmentFile=/opt/ecv-dashboard/.env
ExecStart=/opt/ecv-dashboard/venv/bin/gunicorn -c gunicorn_config.py app:server
ExecReload=/bin/kill -s HUP $MAINPID
Restart=on-failure
RestartSec=10
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

```bash
# Dar permisos
sudo chown -R www-data:www-data /opt/ecv-dashboard

# Habilitar y arrancar servicio
sudo systemctl daemon-reload
sudo systemctl enable ecv-dashboard
sudo systemctl start ecv-dashboard

# Verificar estado
sudo systemctl status ecv-dashboard

# Ver logs
sudo journalctl -u ecv-dashboard -f
```

## 6. Configurar Nginx como Reverse Proxy

```bash
sudo nano /etc/nginx/sites-available/ecv-dashboard
```

Contenido:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # O la IP del servidor

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://127.0.0.1:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (para Dash callbacks)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeout alto para SHAP computations
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
    }

    # Assets estáticos
    location /assets/ {
        alias /opt/ecv-dashboard/assets/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # Límite de tamaño de request
    client_max_body_size 10M;
}
```

```bash
# Activar sitio
sudo ln -s /etc/nginx/sites-available/ecv-dashboard /etc/nginx/sites-enabled/

# Verificar configuración
sudo nginx -t

# Reiniciar Nginx
sudo systemctl restart nginx
```

## 7. Configurar HTTPS (Opcional pero Recomendado)

```bash
# Instalar Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtener certificado SSL
sudo certbot --nginx -d your-domain.com

# Renovación automática
sudo systemctl enable certbot.timer
```

## 8. Configuración Recomendada de Gunicorn

Para un servidor con 4 CPU cores y 8GB RAM:

```python
# gunicorn_config.py optimizado
bind = "0.0.0.0:8050"
workers = 4                    # 2 * CPU + 1, pero limitado por RAM con SHAP
worker_class = "sync"          # sync es más seguro con SHAP/sklearn
timeout = 120                  # Alto para cálculos SHAP
keepalive = 5
preload_app = True             # Comparte modelo en memoria entre workers
max_requests = 1000            # Reinicia workers cada 1000 requests
max_requests_jitter = 50       # Evita reinicio simultáneo
accesslog = "/var/log/ecv-dashboard/access.log"
errorlog = "/var/log/ecv-dashboard/error.log"
loglevel = "warning"
```

```bash
# Crear directorio de logs
sudo mkdir -p /var/log/ecv-dashboard
sudo chown www-data:www-data /var/log/ecv-dashboard
```

## 9. Comandos Útiles

```bash
# Reiniciar el dashboard
sudo systemctl restart ecv-dashboard

# Ver logs en tiempo real
sudo journalctl -u ecv-dashboard -f

# Verificar que el puerto está escuchando
sudo ss -tlnp | grep 8050

# Ver uso de memoria
sudo systemctl status ecv-dashboard --no-pager

# Actualizar el dashboard
cd /opt/ecv-dashboard
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart ecv-dashboard
```

## 10. Monitoreo

Para monitoreo continuo, considere agregar:

```bash
# Instalar htop para monitoreo de recursos
sudo apt install -y htop

# Crear script de healthcheck
cat > /opt/ecv-dashboard/healthcheck.sh << 'EOF'
#!/bin/bash
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8050)
if [ "$HTTP_CODE" -ne 200 ]; then
    echo "Dashboard DOWN (HTTP $HTTP_CODE) - $(date)"
    sudo systemctl restart ecv-dashboard
fi
EOF

chmod +x /opt/ecv-dashboard/healthcheck.sh

# Agregar al crontab (verificar cada 5 minutos)
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/ecv-dashboard/healthcheck.sh >> /var/log/ecv-dashboard/healthcheck.log 2>&1") | crontab -
```

---

## Resumen de Comandos Clave

| Acción | Comando |
|--------|---------|
| Instalar dependencias | `pip install -r requirements.txt` |
| Generar datos | `python data/generate_sample_data.py` |
| Ejecutar (desarrollo) | `python app.py` |
| Ejecutar (producción) | `gunicorn -c gunicorn_config.py app:server` |
| Iniciar servicio | `sudo systemctl start ecv-dashboard` |
| Ver logs | `sudo journalctl -u ecv-dashboard -f` |
| Reiniciar | `sudo systemctl restart ecv-dashboard` |
