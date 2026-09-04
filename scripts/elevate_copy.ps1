$src = "C:\Users\hcsme\Desktop\CUDA Translate\build\bin\nvcuda.dll"
$dst = "C:\Windows\System32\nvcuda.dll"
$sh = New-Object -ComObject Shell.Application
$cmd = "/c copy /y `"$src`" `"$dst`""
Write-Host "Invoking elevated copy: cmd.exe $cmd"
$sh.ShellExecute("cmd.exe", $cmd, "", "runas", 1)
Start-Sleep -Seconds 2
