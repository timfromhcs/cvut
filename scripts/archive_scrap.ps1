$backupDir = Join-Path $HOME "Desktop\CUDA_Translate_Scrap_Backup"
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null

Get-ChildItem -Path . -Filter "*.bak" | Move-Item -Destination $backupDir -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "debug_*" | Move-Item -Destination $backupDir -Force -ErrorAction SilentlyContinue

$itemsToArchive = @(
    "chipStar", "clspv", "deps", "DLSS5-Swapper", "envytools", "maxas", "mesa",
    "OptiScaler", "repro", "SPIRV-Tools", "VkFFT", "vuda", "VulkanMemoryAllocator",
    "workspace", "ZLUDA", "clang.cfg", "clang++.cfg", "ptx_isa_9.0.pdf"
)

foreach ($item in $itemsToArchive) {
    if (Test-Path $item) {
        try {
            Move-Item -Path $item -Destination $backupDir -Force -ErrorAction Stop
            Write-Host "Archived: $item"
        } catch {
            Write-Warning "Failed to move $item : $_"
        }
    }
}

if (Test-Path $backupDir) {
    Write-Host "[STAGE_01_CLEANUP] Success: Archived to $backupDir"
    exit 0
} else {
    exit 1
}
