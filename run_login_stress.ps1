param(
    [string]$Host = "http://127.0.0.1:8000",
    [string]$ArtistsPath = "/artists",
    [int]$Users = 100,
    [int]$SpawnRate = 100,
    [string]$RunTime = "30s",

    [string]$BearerToken
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($BearerToken)) {
    throw "Debes pasar -BearerToken para el test de /artists."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $PSScriptRoot "outputs/stress"
$prefix = Join-Path $outDir ("artists_stress_" + $timestamp)

if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$env:ARTISTS_PATH = $ArtistsPath
$env:BEARER_TOKEN = $BearerToken

Write-Host "Ejecutando stress test..."
Write-Host "Endpoint: GET $ArtistsPath"
Write-Host "Host: $Host"
Write-Host "Usuarios: $Users | SpawnRate: $SpawnRate | Duracion: $RunTime"
Write-Host "Reportes: $prefix*"

locust -f "$PSScriptRoot/locustfile_login_stress.py" `
  --host "$Host" `
  --users $Users `
  --spawn-rate $SpawnRate `
  --headless `
  --run-time $RunTime `
  --only-summary `
  --csv "$prefix" `
  --html "$prefix.html"

Write-Host "Listo. Reportes generados en: $outDir"
