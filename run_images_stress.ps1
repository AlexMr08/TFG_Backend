param(
    [string]$TargetHost = "http://127.0.0.1:8000",
    [string]$ImagesPath = "/view",
    [int]$Users = 1000,
    [int]$SpawnRate = 100,
    [string]$RunTime = "40s",

    [string]$BearerToken
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($BearerToken)) {
    throw "Debes pasar un -BearerToken para ejecutar el test de carga del endpoint /images."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $PSScriptRoot "outputs/stress"
$prefix = Join-Path $outDir ("images_stress_" + $timestamp)

if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$env:IMAGES_PATH = $ImagesPath
$env:BEARER_TOKEN = $BearerToken

Write-Host "Ejecutando test de carga para el endpoint de imagenes..."
Write-Host "Endpoint: GET $ImagesPath"
Write-Host "Host: $TargetHost"
Write-Host "Usuarios: $Users | SpawnRate: $SpawnRate | Duracion: $RunTime"
Write-Host "Reportes: $prefix*"

locust -f "$PSScriptRoot/locustfile_images_stress.py" `
  --host "$TargetHost" `
  --users $Users `
  --spawn-rate $SpawnRate `
  --headless `
  --run-time $RunTime `
  --only-summary `
  --csv "$prefix" `
  --html "$prefix.html"

Write-Host "Listo. Reportes generados en: $outDir"
