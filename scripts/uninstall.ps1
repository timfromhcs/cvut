param (
    [string]$Prefix = "$env:LOCALAPPDATA\CUDA_Vulkan"
)

$ErrorActionPreference = "Stop"

Write-Host "Uninstalling CUDA-to-Vulkan Universal Translator from $Prefix..."

$ManifestPath = Join-Path $Prefix "share\cuda-vulkan\install_manifest.txt"

if (Test-Path $ManifestPath) {
    $files = Get-Content $ManifestPath
    foreach ($file in $files) {
        if (Test-Path $file) {
            Remove-Item -Path $file -Force -ErrorAction SilentlyContinue
        }
    }
}

$dirsToClean = @(
    (Join-Path $Prefix "share\cuda-vulkan\fixtures"),
    (Join-Path $Prefix "share\cuda-vulkan\shaders"),
    (Join-Path $Prefix "share\cuda-vulkan"),
    (Join-Path $Prefix "share"),
    (Join-Path $Prefix "include\cuda"),
    (Join-Path $Prefix "include"),
    (Join-Path $Prefix "lib"),
    (Join-Path $Prefix "bin"),
    $Prefix
)

foreach ($dir in $dirsToClean) {
    if (Test-Path $dir) {
        $items = Get-ChildItem -Path $dir -Force -ErrorAction SilentlyContinue
        if ($null -eq $items -or $items.Count -eq 0) {
            Remove-Item -Path $dir -Force -Recurse -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "Windows uninstallation successfully completed from $Prefix."
