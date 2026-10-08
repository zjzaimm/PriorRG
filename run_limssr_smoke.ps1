$ErrorActionPreference = 'Stop'
$env:HF_HOME = 'D:\codex-cache\huggingface'
$env:PIP_CACHE_DIR = 'D:\codex-cache\pip'
$env:TEMP = 'D:\codex-cache\tmp'
$env:TMP = $env:TEMP
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$taskPython = 'D:\codex-envs\priorrg-limssr\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw "Virtual environment not found: $taskPython"
}
& $taskPython smoke_test_limssr.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -m unittest models.limssr.test_modules -v
exit $LASTEXITCODE
