param (
    [string]$Prefix = "$env:LOCALAPPDATA\CUDA_Vulkan"
)

$ErrorActionPreference = "Stop"

Write-Host "Installing CUDA-to-Vulkan Universal Translator to $Prefix..."

$RootDir = Split-Path -Parent $PSScriptRoot
$DistDir = Join-Path $RootDir "dist"

$BinDir = Join-Path $Prefix "bin"
$LibDir = Join-Path $Prefix "lib"
$IncDir = Join-Path $Prefix "include\cuda"
$ShaderDir = Join-Path $Prefix "share\cuda-vulkan\shaders"
$FixtureDir = Join-Path $Prefix "share\cuda-vulkan\fixtures"

New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
New-Item -ItemType Directory -Force -Path $LibDir | Out-Null
New-Item -ItemType Directory -Force -Path $IncDir | Out-Null
New-Item -ItemType Directory -Force -Path $ShaderDir | Out-Null
New-Item -ItemType Directory -Force -Path $FixtureDir | Out-Null

$ManifestPath = Join-Path $Prefix "share\cuda-vulkan\install_manifest.txt"
$ManifestList = [System.Collections.Generic.List[string]]::new()

function Copy-FilesWithManifest($SrcPattern, $DestDir) {
    if (Test-Path $SrcPattern) {
        Get-ChildItem -Path $SrcPattern -File | ForEach-Object {
            $destFile = Join-Path $DestDir $_.Name
            Copy-Item -Path $_.FullName -Destination $destFile -Force
            $ManifestList.Add($destFile)
        }
    }
}

Copy-FilesWithManifest (Join-Path $DistDir "bin\*") $BinDir
Copy-FilesWithManifest (Join-Path $DistDir "lib\*") $LibDir
Copy-FilesWithManifest (Join-Path $DistDir "include\*") $IncDir
Copy-FilesWithManifest (Join-Path $DistDir "shaders\*") $ShaderDir
Copy-FilesWithManifest (Join-Path $DistDir "fixtures\*") $FixtureDir

$ManifestList.Add($ManifestPath)
Set-Content -Path $ManifestPath -Value $ManifestList

Write-Host "Windows installation successfully completed at $Prefix."
