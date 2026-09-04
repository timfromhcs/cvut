param (
    [switch]$VerifyOnly,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$NvsmiDir = "C:\Program Files\NVIDIA Corporation\NVSMI"
$CudaVkBin = "C:\CUDA_Vulkan\bin"
$CudaVkLib = "C:\CUDA_Vulkan\lib"
$CudaVkInc = "C:\CUDA_Vulkan\include"
$System32Dir = "C:\Windows\System32"
$UserBin = "C:\Users\hcsme\bin"

if ($VerifyOnly) {
    Write-Host "[DEPLOY_VERIFY] Checking system deployment status..."

    $missing = 0

    $smiFound = (Test-Path (Join-Path $NvsmiDir "nvidia-smi.exe")) -or (Test-Path (Join-Path $CudaVkBin "nvidia-smi.exe"))
    $nvmlFound = (Test-Path (Join-Path $NvsmiDir "nvml.dll")) -or (Test-Path (Join-Path $CudaVkBin "nvml.dll"))
    $cudartFound = Test-Path (Join-Path $CudaVkBin "cudart64_12.dll")
    $nvcudaFound = Test-Path (Join-Path $CudaVkBin "nvcuda.dll")

    if (-not $smiFound) {
        Write-Warning "Missing nvidia-smi.exe in deployment directories"
        $missing++
    } else {
        Write-Host "  Verified: nvidia-smi.exe deployed"
    }

    if (-not $nvmlFound) {
        Write-Warning "Missing nvml.dll in deployment directories"
        $missing++
    } else {
        Write-Host "  Verified: nvml.dll deployed"
    }

    if (-not $cudartFound) {
        Write-Warning "Missing cudart64_12.dll in $CudaVkBin"
        $missing++
    } else {
        Write-Host "  Verified: cudart64_12.dll deployed"
    }

    if (-not $nvcudaFound) {
        Write-Warning "Missing nvcuda.dll in $CudaVkBin"
        $missing++
    } else {
        Write-Host "  Verified: nvcuda.dll deployed"
    }

    $combinedPath = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $env:Path

    $pathVerified = ($combinedPath -like "*$CudaVkBin*") -or ($combinedPath -like "*$NvsmiDir*") -or ($combinedPath -like "*$UserBin*")
    if (-not $pathVerified) {
        Write-Warning "Deployment directory not found in PATH"
        $missing++
    } else {
        Write-Host "  Verified: Deployment directories present in PATH"
    }

    if ($missing -eq 0) {
        Write-Host "[DEPLOY_VERIFY] SUCCESS: All deployment targets verified."
        exit 0
    } else {
        Write-Error "[DEPLOY_VERIFY] FAILED: $missing checks failed."
        exit 1
    }
}

if ($Uninstall) {
    Write-Host "[DEPLOY] Uninstalling system deployment..."
    Remove-Item -Path (Join-Path $NvsmiDir "nvidia-smi.exe") -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $NvsmiDir "nvml.dll") -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $CudaVkBin "cudart64_12.dll") -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $CudaVkBin "nvcuda.dll") -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $CudaVkBin "nvidia-smi.exe") -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $CudaVkBin "nvml.dll") -Force -ErrorAction SilentlyContinue
    Write-Host "[DEPLOY] Uninstallation complete."
    exit 0
}

Write-Host "[DEPLOY] 1. Creating deployment directories..."
New-Item -ItemType Directory -Force -Path $CudaVkBin | Out-Null
New-Item -ItemType Directory -Force -Path $CudaVkLib | Out-Null
New-Item -ItemType Directory -Force -Path $CudaVkInc | Out-Null
New-Item -ItemType Directory -Force -Path $UserBin | Out-Null

$nvsmiWritable = $false
try {
    New-Item -ItemType Directory -Force -Path $NvsmiDir -ErrorAction Stop | Out-Null
    $nvsmiWritable = $true
} catch {
    Write-Host "  Note: Program Files requires elevation; deploying NVSMI to $CudaVkBin and $UserBin."
}

Write-Host "[DEPLOY] 2. Copying nvidia-smi.exe and nvml.dll..."
Copy-Item -Path (Join-Path $RootDir "build\bin\nvidia-smi.exe") -Destination (Join-Path $CudaVkBin "nvidia-smi.exe") -Force
Copy-Item -Path (Join-Path $RootDir "build\bin\nvml.dll") -Destination (Join-Path $CudaVkBin "nvml.dll") -Force
Copy-Item -Path (Join-Path $RootDir "build\bin\nvidia-smi.exe") -Destination (Join-Path $UserBin "nvidia-smi.exe") -Force
Copy-Item -Path (Join-Path $RootDir "build\bin\nvml.dll") -Destination (Join-Path $UserBin "nvml.dll") -Force

if ($nvsmiWritable) {
    Copy-Item -Path (Join-Path $RootDir "build\bin\nvidia-smi.exe") -Destination (Join-Path $NvsmiDir "nvidia-smi.exe") -Force -ErrorAction SilentlyContinue
    Copy-Item -Path (Join-Path $RootDir "build\bin\nvml.dll") -Destination (Join-Path $NvsmiDir "nvml.dll") -Force -ErrorAction SilentlyContinue
}

Write-Host "[DEPLOY] 3. Copying runtime libraries to $CudaVkBin..."
Copy-Item -Path (Join-Path $RootDir "build\lib\cudart64_12.dll") -Destination (Join-Path $CudaVkBin "cudart64_12.dll") -Force
Copy-Item -Path (Join-Path $RootDir "build\lib\nvcuda.dll") -Destination (Join-Path $CudaVkBin "nvcuda.dll") -Force
Copy-Item -Path (Join-Path $RootDir "build\lib\cudart64_12.dll") -Destination (Join-Path $CudaVkLib "cudart64_12.dll") -Force
Copy-Item -Path (Join-Path $RootDir "build\lib\nvcuda.dll") -Destination (Join-Path $CudaVkLib "nvcuda.dll") -Force
Copy-Item -Path (Join-Path $RootDir "build\lib\*.lib") -Destination $CudaVkBin -Force -ErrorAction SilentlyContinue
Copy-Item -Path (Join-Path $RootDir "build\lib\*.lib") -Destination $CudaVkLib -Force -ErrorAction SilentlyContinue

Copy-Item -Path (Join-Path $RootDir "build\lib\cudart64_12.dll") -Destination (Join-Path $UserBin "cudart64_12.dll") -Force -ErrorAction SilentlyContinue
Copy-Item -Path (Join-Path $RootDir "build\lib\nvcuda.dll") -Destination (Join-Path $UserBin "nvcuda.dll") -Force -ErrorAction SilentlyContinue

Write-Host "[DEPLOY] 4. Copying headers to $CudaVkInc..."
Copy-Item -Path (Join-Path $RootDir "src\runtime\*.h") -Destination $CudaVkInc -Force -ErrorAction SilentlyContinue

Write-Host "[DEPLOY] 5. Placing nvml.dll and nvcuda.dll into $System32Dir..."
try {
    $sysNvml = Join-Path $System32Dir "nvml.dll"
    if (Test-Path $sysNvml) {
        Copy-Item -Path $sysNvml -Destination "$sysNvml.bak" -Force -ErrorAction SilentlyContinue
    }
    Copy-Item -Path (Join-Path $RootDir "build\bin\nvml.dll") -Destination $sysNvml -Force -ErrorAction SilentlyContinue

    $sysNvcuda = Join-Path $System32Dir "nvcuda.dll"
    if (Test-Path $sysNvcuda) {
        Copy-Item -Path $sysNvcuda -Destination "$sysNvcuda.bak" -Force -ErrorAction SilentlyContinue
    }
    Copy-Item -Path (Join-Path $RootDir "build\lib\nvcuda.dll") -Destination $sysNvcuda -Force -ErrorAction SilentlyContinue
} catch {
    Write-Host "  Note: System32 requires elevation; binaries available via system PATH."
}

Write-Host "[DEPLOY] 6. Updating PATH environment variable..."
function Add-ToPath($targetDir) {
    try {
        $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($currentPath -notlike "*$targetDir*") {
            $newPath = "$targetDir;$currentPath"
            [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
            Write-Host "  Added to User PATH: $targetDir"
        }
    } catch {}

    try {
        $machPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        if ($machPath -notlike "*$targetDir*") {
            [Environment]::SetEnvironmentVariable("Path", "$targetDir;$machPath", "Machine")
            Write-Host "  Added to Machine PATH: $targetDir"
        }
    } catch {}

    if ($env:Path -notlike "*$targetDir*") {
        $env:Path = "$targetDir;$env:Path"
    }
}

Add-ToPath $CudaVkBin
Add-ToPath $UserBin
if ($nvsmiWritable) {
    Add-ToPath $NvsmiDir
}

Write-Host "[DEPLOY] System deployment completed successfully."
