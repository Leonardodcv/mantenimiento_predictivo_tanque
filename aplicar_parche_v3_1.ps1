param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectPath
)

$ErrorActionPreference = "Stop"
$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectPath = (Resolve-Path $ProjectPath).Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupRoot = Join-Path $ProjectPath ("backup_v3_0_" + $Timestamp)

Write-Host "Proyecto: $ProjectPath"
Write-Host "Respaldo: $BackupRoot"
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null

$Files = @(
    "app\config.py",
    "app\main.py",
    "app\services\model_manager.py",
    "app\services\anomaly_engine_v31.py",
    "tests\test_engine_v31.py",
    "README.md",
    "CHANGELOG_V3_1.md"
)

foreach ($Relative in $Files) {
    $Target = Join-Path $ProjectPath $Relative
    if (Test-Path $Target) {
        $Backup = Join-Path $BackupRoot $Relative
        $BackupDir = Split-Path -Parent $Backup
        New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
        Copy-Item -Force $Target $Backup
    }
}

$CopyMap = @{
    "app\config.py" = "app\config.py"
    "app\main.py" = "app\main.py"
    "app\services\model_manager.py" = "app\services\model_manager.py"
    "app\services\anomaly_engine_v31.py" = "app\services\anomaly_engine_v31.py"
    "tests\test_engine_v31.py" = "tests\test_engine_v31.py"
    "README_V3_1.md" = "README.md"
    "CHANGELOG_V3_1.md" = "CHANGELOG_V3_1.md"
}

foreach ($SourceRelative in $CopyMap.Keys) {
    $Source = Join-Path $PatchRoot $SourceRelative
    $Target = Join-Path $ProjectPath $CopyMap[$SourceRelative]
    $TargetDir = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    Copy-Item -Force $Source $Target
}

# .env.example es referencia; se actualiza. El .env real NO se toca.
$EnvExampleSource = Join-Path $PatchRoot ".env.example"
$EnvExampleTarget = Join-Path $ProjectPath ".env.example"
if (Test-Path $EnvExampleTarget) {
    Copy-Item -Force $EnvExampleTarget (Join-Path $BackupRoot ".env.example")
}
Copy-Item -Force $EnvExampleSource $EnvExampleTarget

Write-Host ""
Write-Host "Parche v3.1 aplicado correctamente."
Write-Host "Tu archivo .env no fue modificado."
Write-Host "Ejecuta: python -m pytest -q"
Write-Host "Luego reinicia con: python run.py"
Write-Host "Finalmente ejecuta POST /api/v3/model/rebuild/?source=sqlserver"
