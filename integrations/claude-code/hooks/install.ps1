# Install the `map` binary of the newest release into DIR, with the install
# script of the release. The session-start hook gives the model this command
# when no binary exists. A script file with plain arguments runs the same
# from the Bash tool and from the PowerShell tool: a one-line command with
# quotes inside quotes does not.
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 DIR VARIANT
#
# VARIANT is `default` or `full`.
param(
    [Parameter(Mandatory = $true)] [string] $Dir,
    [string] $Variant = 'default'
)
$ErrorActionPreference = 'Stop'
$env:MAP_INSTALL_DIR = $Dir
$env:MAP_INSTALL_VARIANT = $Variant
Invoke-RestMethod https://github.com/hanpa-nai/map/releases/latest/download/install.ps1 | Invoke-Expression
