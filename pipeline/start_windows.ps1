# SPDX-License-Identifier: AGPL-3.0-only

# Launched by start_windows.bat using the Windows copy of PowerShell.
$ErrorActionPreference = 'Stop'
$appArguments = $args
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')

function Test-AppPython {
    param([string]$Python, [string[]]$Check)

    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { return $false }
    $ErrorActionPreference = 'SilentlyContinue'
    & $Python -B @Check *> $null
    return $?
}

function Confirm-AppSetup {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()

    $dialog = [System.Windows.Forms.Form]::new()
    try {
        $dialog.Text = 'Set up CanopySplit Studio'
        $dialog.StartPosition = 'CenterScreen'
        $dialog.FormBorderStyle = 'FixedDialog'
        $dialog.MinimizeBox = $false
        $dialog.MaximizeBox = $false
        $dialog.TopMost = $true
        $dialog.AutoScaleMode = 'Dpi'
        $dialog.AutoScaleDimensions = [System.Drawing.SizeF]::new(96, 96)
        $dialog.ClientSize = [System.Drawing.Size]::new(460, 164)

        $message = [System.Windows.Forms.Label]::new()
        $message.Location = [System.Drawing.Point]::new(18, 18)
        $message.Size = [System.Drawing.Size]::new(424, 92)
        $message.Text = ('CanopySplit Studio needs Python or required libraries before it can open.' +
            "`r`n`r`n" + 'Choose Install to download the required software into the CanopySplit folder. Setup needs an internet connection.')

        $cancel = [System.Windows.Forms.Button]::new()
        $cancel.Text = 'Cancel'
        $cancel.Location = [System.Drawing.Point]::new(262, 116)
        $cancel.Size = [System.Drawing.Size]::new(84, 30)
        $cancel.DialogResult = 'Cancel'

        $install = [System.Windows.Forms.Button]::new()
        $install.Text = 'Install'
        $install.Location = [System.Drawing.Point]::new(358, 116)
        $install.Size = [System.Drawing.Size]::new(84, 30)
        $install.DialogResult = 'OK'

        $dialog.Controls.AddRange(@($message, $cancel, $install))
        $dialog.CancelButton = $cancel
        $dialog.AcceptButton = $cancel
        $dialog.ActiveControl = $cancel
        return ($dialog.ShowDialog() -eq 'OK')
    }
    finally {
        $dialog.Dispose()
    }
}

function Invoke-SetupUv {
    param([string]$Uv)

    & $Uv @args
    if ($LASTEXITCODE -ne 0) { throw "Software setup failed during uv $($args[0])." }
}

function Install-AppRequirements {
    param([string]$Python)

    $appData = Join-Path (Get-Location).Path '.canopysplit'
    $setupDir = Join-Path $appData ('setup-' + [Guid]::NewGuid())
    $null = New-Item -ItemType Directory -Path $setupDir -Force
    try {
        # Verify the official release archive before running its executable.
        $uvVersion = '0.13.9'
        $uvTarget = 'uv-x86_64-pc-windows-msvc.zip'
        $uvSha256 = 'edd61f7f5728a56cc62455386661860bdfc7bab65872f39b0c25bdcc1d8e1beb'
        $archive = Join-Path $setupDir $uvTarget
        $ProgressPreference = 'SilentlyContinue'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 90 -OutFile $archive `
            -Uri "https://github.com/astral-sh/uv/releases/download/$uvVersion/$uvTarget"
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $uvSha256) {
            throw 'The setup download could not be verified. Please open the launcher again to retry.'
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $setupDir
        $uv = Join-Path $setupDir 'uv.exe'
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $appData 'python'
        $env:UV_CACHE_DIR = Join-Path $appData 'uv-cache'
        $env:UV_PYTHON_INSTALL_REGISTRY = '0'
        $env:UV_PYTHON_INSTALL_BIN = '0'

        $pythonCheck = @('-c', 'import sys, tkinter; sys.exit(sys.version_info < (3, 10))')
        if (-not (Test-AppPython -Python $Python -Check $pythonCheck)) {
            Invoke-SetupUv $uv venv --no-config --allow-existing --managed-python --python 3.11 .venv
            if (-not (Test-AppPython -Python $Python -Check $pythonCheck)) {
                throw 'The local Python installation could not start with folder browsing support.'
            }
        }
        Invoke-SetupUv $uv pip install --no-config --only-binary ':all:' --torch-backend cpu --python $Python -r requirements.txt
        Invoke-SetupUv $uv pip check --no-config --python $Python
        & $Python -B -m pipeline.readiness installed
        if ($LASTEXITCODE -ne 0) { throw 'Some required libraries are missing after setup.' }
    }
    finally {
        Remove-Item -LiteralPath $setupDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

try {
    Write-Host 'Checking CanopySplit Studio setup on this PC.'
    if (-not ((Test-Path -LiteralPath 'requirements.txt' -PathType Leaf) -and
              (Test-Path -LiteralPath 'app.py' -PathType Leaf))) {
        throw 'Extract all files from the CanopySplit ZIP before opening the launcher.'
    }

    $localPython = Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
    $readyCheck = @('-m', 'pipeline.readiness', 'installed')
    $appPython = $null
    if (Test-AppPython -Python $localPython -Check $readyCheck) {
        $appPython = $localPython
    }
    else {
        # Skip the Microsoft Store placeholder for Python on a fresh PC.
        $systemPythons = Get-Command python.exe -CommandType Application -All -ErrorAction SilentlyContinue |
            Where-Object { $_.Source -notlike "$env:LOCALAPPDATA\Microsoft\WindowsApps\*" }
        foreach ($candidate in $systemPythons) {
            if (Test-AppPython -Python $candidate.Source -Check $readyCheck) {
                $appPython = $candidate.Source
                break
            }
        }
    }

    if (-not $appPython) {
        if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
            throw 'Automatic setup requires 64-bit Windows on an Intel or AMD PC.'
        }
        if (-not (Confirm-AppSetup)) {
            Write-Host 'Setup canceled. Open start_windows.bat again whenever you are ready.'
            exit 0
        }

        $null = New-Item -ItemType Directory -Path '.venv' -Force
        $lockPath = Join-Path (Get-Location).Path '.venv\canopysplit-setup.lock'
        try {
            $setupLock = [System.IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None')
        }
        catch [System.IO.IOException] {
            throw 'Setup is already running or the app folder is busy. Close any other setup window and try again.'
        }
        try {
            if (-not (Test-AppPython -Python $localPython -Check $readyCheck)) {
                Write-Host 'Setting up CanopySplit Studio. Keep this window open while the required software is installed.'
                Install-AppRequirements -Python $localPython
            }
        }
        finally {
            $setupLock.Dispose()
        }
        $appPython = $localPython
    }

    Write-Host 'Opening CanopySplit Studio in your browser. Keep this window open while you work.'
    & $appPython -B -u app.py --port 0 @appArguments
    exit $LASTEXITCODE
}
catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
