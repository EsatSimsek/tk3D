[CmdletBinding()]
param(
    [string]$Session = "data\\aist_test\\session.yaml",
    [string]$OutputRoot = "outputs"
)

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom
chcp 65001 | Out-Null

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv312\\Scripts\\python.exe"
$pipelinePath = Join-Path $projectRoot "scripts\\run_vitpose_multiview_3d.py"
$sessionPath = Join-Path $projectRoot $Session
$outputPath = Join-Path $projectRoot $OutputRoot

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python ortamı bulunamadı: $pythonPath"
}
if (-not (Test-Path -LiteralPath $sessionPath -PathType Leaf)) {
    throw "Oturum dosyası bulunamadı: $sessionPath"
}

$runId = "pose_" + (Get-Date -Format "yyyy-MM-dd_HH-mm-ss_fff")
$sessionOutputRoot = Join-Path $outputPath "aist_test"
$runOutputPath = Join-Path (Join-Path $sessionOutputRoot "runs") $runId
Write-Host "Yeni çıktı klasörü: $runOutputPath"

& $pythonPath $pipelinePath `
    --session $sessionPath `
    --output-root $outputPath `
    --stride 1 `
    --progress-every 20 `
    --run-id $runId

if ($LASTEXITCODE -ne 0) {
    throw "İşlem başarısız oldu. Çıkış kodu: $LASTEXITCODE"
}

Write-Host "Tamamlanan videolar: $(Join-Path $runOutputPath 'videos')"
