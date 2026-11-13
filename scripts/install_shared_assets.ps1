#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackageRoot,

    [switch]$Force,

    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$packagePath = (Resolve-Path -LiteralPath $PackageRoot).Path
$repositoryPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$userProfilePath = [Environment]::GetFolderPath("UserProfile")

$assets = @(
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\videos\zed_35151067.avi"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\videos\zed_35151067.avi"
    },
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\videos\zed_37137479.avi"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\videos\zed_37137479.avi"
    },
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\svo\ZED-2i_SN35151067_2026-07-31_18-31-08.svo2"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\svo\ZED-2i_SN35151067_2026-07-31_18-31-08.svo2"
    },
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\svo\ZED-2i_SN37137479_2026-07-31_18-31-08.svo2"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\svo\ZED-2i_SN37137479_2026-07-31_18-31-08.svo2"
    },
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\reports\zed_35151067_timestamp_mapping.json"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\reports\zed_35151067_timestamp_mapping.json"
    },
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\reports\zed_37137479_timestamp_mapping.json"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\reports\zed_37137479_timestamp_mapping.json"
    },
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\reference\json\vitpose_session_3d.json"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\reference\json\vitpose_session_3d.json"
    },
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\reference\csv\vitpose_keypoints_2d_flat.csv"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\reference\csv\vitpose_keypoints_2d_flat.csv"
    },
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\reference\refined_20261113\vitpose_session_3d.json"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\reference\refined_20261113\vitpose_session_3d.json"
    },
    @{
        Package = "repo\data\poomsae_1_zed2i_20260731_trimmed\reference\refined_20261113\run_manifest.json"
        Destination = Join-Path $repositoryPath "data\poomsae_1_zed2i_20260731_trimmed\reference\refined_20261113\run_manifest.json"
    },
    @{
        Package = "repo\outputs\poomsae_1_zed2i_20260731_trimmed\calibration\cameras.json"
        Destination = Join-Path $repositoryPath "outputs\poomsae_1_zed2i_20260731_trimmed\calibration\cameras.json"
    },
    @{
        Package = "repo\outputs\poomsae_1_zed2i_20260731_trimmed\calibration\cameras_keypoint_refined_20261113.json"
        Destination = Join-Path $repositoryPath "outputs\poomsae_1_zed2i_20260731_trimmed\calibration\cameras_keypoint_refined_20261113.json"
    },
    @{
        Package = "repo\weights\vitpose_huge_wholebody_256x192.pth"
        Destination = Join-Path $repositoryPath "weights\vitpose_huge_wholebody_256x192.pth"
    },
    @{
        Package = "user_home\.roboflow\models\rf-detr-small.pth"
        Destination = Join-Path $userProfilePath ".roboflow\models\rf-detr-small.pth"
    }
)

$manifestPath = Join-Path $packagePath "SHA256SUMS.txt"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Package checksum file was not found: $manifestPath"
}

$manifest = @{}
foreach ($line in Get-Content -LiteralPath $manifestPath -Encoding UTF8) {
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }
    if ($line -notmatch '^([0-9a-fA-F]{64})\s+\*(.+)$') {
        throw "Invalid SHA256SUMS line: $line"
    }
    $manifest[$Matches[2].Replace("/", "\")] = $Matches[1].ToLowerInvariant()
}

function Get-Sha256Lower([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

foreach ($asset in $assets) {
    $source = Join-Path $packagePath $asset.Package
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Package file is missing: $source"
    }
    if (-not $manifest.ContainsKey($asset.Package)) {
        throw "Package file has no SHA-256 entry: $($asset.Package)"
    }
    $actualHash = Get-Sha256Lower $source
    if ($actualHash -ne $manifest[$asset.Package]) {
        throw "Package file is corrupt or modified: $($asset.Package)"
    }
    Write-Host "VERIFIED    $($asset.Package)"
}

$fusionCalibration = "extras\zed_fusion_calibration.json"
$fusionSource = Join-Path $packagePath $fusionCalibration
if (-not (Test-Path -LiteralPath $fusionSource -PathType Leaf)) {
    throw "ZED Fusion calibration is missing: $fusionSource"
}
if (-not $manifest.ContainsKey($fusionCalibration) -or
    (Get-Sha256Lower $fusionSource) -ne $manifest[$fusionCalibration]) {
    throw "ZED Fusion calibration failed SHA-256 verification."
}
Write-Host "VERIFIED    $fusionCalibration"

if ($VerifyOnly) {
    Write-Host "Package is complete and SHA-256 verification passed. No files were copied."
    exit 0
}

foreach ($asset in $assets) {
    $source = Join-Path $packagePath $asset.Package
    $destination = $asset.Destination
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null

    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        if ((Get-Sha256Lower $destination) -eq (Get-Sha256Lower $source)) {
            Write-Host "ALREADY OK  $destination"
            continue
        }
        if (-not $Force) {
            throw "A different destination file already exists: $destination (use -Force to overwrite)"
        }
    }

    Copy-Item -LiteralPath $source -Destination $destination -Force:$Force
    if ((Get-Sha256Lower $destination) -ne (Get-Sha256Lower $source)) {
        throw "SHA-256 mismatch after copy: $destination"
    }
    Write-Host "INSTALLED   $destination"
}

Write-Host "TK3D external assets are installed. Next step: tk3d-check"
