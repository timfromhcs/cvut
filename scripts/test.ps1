# SPDX-License-Identifier: Apache-2.0
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& python "$ScriptDir\test.py" @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
