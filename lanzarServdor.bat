@echo off
title Lanzador de Servidores TFG - Alex

echo Lanzando vLLM en WSL (Puerto 8001)...
start "vLLM-Server" wsl bash -c "source /home/alex/tfg/venv_tfg/bin/activate && vllm serve Qwen/Qwen2-VL-7B-Instruct-AWQ --quantization awq_marlin --dtype half --gpu-memory-utilization 0.7 --max-model-len 8192 --port 8002 --host 0.0.0.0"

timeout /t 45

echo Lanzando FastAPI (Puerto 8000)...
start "FastAPI-Backend" uvicorn servidorFA:app --host localhost --port 8000

echo Abriendo tunel de Ngrok...
start "Ngrok-Tunnel" ngrok http 8000

echo.
echo Todos los servicios estan iniciados. Pero puede que tarden unos segundos en estar operativos.
pause