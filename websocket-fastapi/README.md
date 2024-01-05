# Real time STT using RecordRTC

## how-to

1. Install dependencies,

```bash
pip3 install -r requirements.txt
```

2. Run FastAPI,

```bash
IMPORT_LOCAL=true uvicorn app.main:app --reload --host 0.0.0.0
```

Or use docker,

```bash
docker-compose up --build
```

## push to dockerhub

```bash
docker build -t mesoliticadev/realtime-stt-websocket .
docker push mesoliticadev/realtime-stt-websocket
```