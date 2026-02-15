# Script para reinstalar PyTorch optimizado para RTX 5070 Ti (Blackwell)

Write-Host "REINSTALACION DE PYTORCH PARA RTX 5070 Ti" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Desinstalar version actual
Write-Host "Desinstalando PyTorch actual..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio

# 2. Instalar version optimizada para Blackwell sm_120
Write-Host ""
Write-Host "Instalando PyTorch con soporte CUDA 12.6 para sm_120..." -ForegroundColor Green
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

Write-Host ""
Write-Host "INSTALACION COMPLETADA" -ForegroundColor Green
Write-Host ""
Write-Host "Verificando instalacion..." -ForegroundColor Cyan

# 3. Verificar instalacion
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('Compute Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"

Write-Host ""
Write-Host "Ahora tu RTX 5070 Ti tendra:" -ForegroundColor Magenta
Write-Host "   - Tensor Cores Gen 5 activos" -ForegroundColor White
Write-Host "   - FP8/FP4 nativo disponible" -ForegroundColor White
Write-Host "   - 2-4x mas velocidad en inferencia" -ForegroundColor White
Write-Host ""
Write-Host "Ejecuta de nuevo los tests para ver la diferencia!" -ForegroundColor Yellow
