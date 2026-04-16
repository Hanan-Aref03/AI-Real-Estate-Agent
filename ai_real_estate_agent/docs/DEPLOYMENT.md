# Deployment Guide

## Deployment Options

### 1. Local Development

Perfect for testing and development.

```bash
# Install dependencies
uv sync

# Run FastAPI server
uv run python -m app.main

# In another terminal, run Streamlit
uv run streamlit run ui/streamlit_app.py
```

### 2. Docker Container

Best for isolated, reproducible environments.

#### Build

```bash
docker build -t ai-real-estate-agent:latest .
```

#### Run

```bash
docker run -d \
  --name real-estate-agent \
  -p 8000:8000 \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
  -v $(pwd)/models:/app/models \
  ai-real-estate-agent:latest
```

#### View Logs

```bash
docker logs real-estate-agent
```

#### Stop Container

```bash
docker stop real-estate-agent
docker rm real-estate-agent
```

### 3. Production Deployment

#### Using Gunicorn

For higher performance with multiple workers:

```bash
pip install gunicorn

gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

#### Using systemd

Create `/etc/systemd/system/real-estate-agent.service`:

```ini
[Unit]
Description=AI Real Estate Agent
After=network.target

[Service]
Type=simple
User=webapp
WorkingDirectory=/opt/ai-real-estate-agent
Environment="PATH=/opt/ai-real-estate-agent/.venv/bin"
ExecStart=/opt/ai-real-estate-agent/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl start real-estate-agent
sudo systemctl enable real-estate-agent
```

### 4. Cloud Deployment

#### AWS

**Option A: EC2**
1. Launch Ubuntu instance
2. Install Python 3.11, git, uv
3. Clone repository
4. Run with systemd (see above)
5. Set up nginx as reverse proxy
6. Configure security groups

**Option B: ECS (Docker)**
1. Push image to ECR
2. Create ECS task definition
3. Launch service
4. Configure load balancer

#### Heroku

Create `Procfile`:

```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Deploy:

```bash
git push heroku main
```

#### Google Cloud Run

```bash
gcloud run deploy real-estate-agent \
  --source . \
  --platform managed \
  --region us-central1
```

## Environment Variables

### Required

- `OPENAI_API_KEY` - OpenAI API key for GPT models
- `ANTHROPIC_API_KEY` - Anthropic API key (optional alternative)

### Optional

- `API_PORT` - Server port (default: 8000)
- `LOG_LEVEL` - Logging level (default: INFO)

## SSL/TLS Setup

### With Nginx

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Logging

Configure logging in `config.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Metrics

Consider integrating Prometheus for metrics:

```bash
pip install prometheus-client
```

## Performance Optimization

1. **Model Caching** - Models loaded once at startup
2. **Connection Pooling** - Reuse LLM client connections
3. **Async Processing** - Use async/await for I/O operations
4. **Database Indexing** - If using database (not in current setup)

## Scaling

### Horizontal Scaling

Deploy multiple container instances behind a load balancer (nginx, AWS ELB).

### Vertical Scaling

- Increase CPU/memory
- Use Gunicorn with more workers
- Implement caching layer (Redis)

## Backup & Recovery

1. **Model Backups** - Store model files in cloud storage
2. **Configuration** - Version control `.env` templates
3. **Disaster Recovery** - Automated container restart policies

## Security

- [ ] (PR) Add API key authentication
- [ ] Enable HTTPS/SSL in production
- [ ] Implement rate limiting
- [ ] Add request validation
- [ ] Regular security updates
- [ ] Environment variable secrets management

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Module Import Errors

```bash
# Ensure dependencies installed
uv sync

# Verify Python path
python -c "import app"
```

### API Key Issues

```bash
# Verify .env is loaded
python -c "from app.config import OPENAI_API_KEY; print(OPENAI_API_KEY)"
```
