<#
.SYNOPSIS
  Install the `map` binary of the newest release. Rust is not necessary.

.DESCRIPTION
      irm https://github.com/hanpa-nai/map/releases/latest/download/install.ps1 | iex

  The script downloads one archive and the file `SHA256SUMS`. It stops if the
  SHA-256 of the archive is not the recorded one. Then it puts `map.exe` in
  the install directory. It does not change `PATH`.

  Environment variables:

    MAP_INSTALL_DIR      The install directory.
                         Default: %LOCALAPPDATA%\Programs\map\bin
    MAP_INSTALL_VARIANT  `full` gets the binary with all features.
                         Default: `default`, the default binary.
    MAP_INSTALL_BASE     The location of the release files: the start of a
                         URL, or a local directory. Default: the newest
                         release on GitHub.
    MAP_INSTALL_TARGET   The target of the binary.
                         Default: x86_64-pc-windows-msvc

  This file is ASCII with no byte order mark, and it runs in Windows
  PowerShell 5.1. The `irm` of that version does not remove a byte order
  mark, and `iex` then stops with a parse error.
#>

# The body is a function, and the last line of the file is the only call.
# Thus a download that stops before the end of the file runs no command. The
# settings of the script also do not stay in the session of the user.
function Install-Map {
    $ErrorActionPreference = 'Stop'
    # The progress bar of Windows PowerShell 5.1 makes a download many times
    # slower. Measured for an archive of 4 MB from a server on the same
    # machine: 3 s to 5 s with the bar, and less than 0.1 s without it.
    $ProgressPreference = 'SilentlyContinue'

    $defaultBase = 'https://github.com/hanpa-nai/map/releases/latest/download'

    if ($env:OS -ne 'Windows_NT') {
        throw "this script is for Windows. On Linux and macOS, run: curl -fsSL $defaultBase/install.sh | sh"
    }

    $base = $env:MAP_INSTALL_BASE
    if (-not $base) { $base = $defaultBase }
    $base = $base.TrimEnd('/', '\')

    $variant = $env:MAP_INSTALL_VARIANT
    if (-not $variant) { $variant = 'default' }
    if ($variant -eq 'default') {
        $prefix = 'map'
    } elseif ($variant -eq 'full') {
        $prefix = 'map-full'
    } else {
        throw "MAP_INSTALL_VARIANT is ``$variant``. Use ``default`` or ``full``."
    }

    $target = $env:MAP_INSTALL_TARGET
    if (-not $target) {
        # The release has one Windows binary, for x64. Windows on ARM64 runs
        # an x64 program. 32-bit Windows does not.
        if (-not [Environment]::Is64BitOperatingSystem) {
            throw 'no binary is available for 32-bit Windows. Build MAP with cargo.'
        }
        $target = 'x86_64-pc-windows-msvc'
    }
    if ($target -notlike '*windows*') {
        throw "the binary for $target is not a Windows program. On Linux and macOS, run: curl -fsSL $defaultBase/install.sh | sh"
    }

    $dir = $env:MAP_INSTALL_DIR
    if (-not $dir) {
        if (-not $env:LOCALAPPDATA) {
            throw 'LOCALAPPDATA is not set. Set MAP_INSTALL_DIR to the install directory.'
        }
        $dir = Join-Path $env:LOCALAPPDATA 'Programs\map\bin'
    }

    $archive = "$prefix-$target.zip"
    $fromUrl = $base -match '^[A-Za-z][A-Za-z0-9+.-]+://'
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ('map-install-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        if ($fromUrl) {
            Write-Host "Downloading $base/$archive"
        } else {
            Write-Host "Copying $(Join-Path $base $archive)"
        }
        foreach ($name in $archive, 'SHA256SUMS') {
            $to = Join-Path $tmp $name
            if ($fromUrl) {
                try {
                    # Without `-UseBasicParsing`, Windows PowerShell 5.1
                    # uses the Internet Explorer engine, and a machine can
                    # have no such engine.
                    Invoke-WebRequest -Uri "$base/$name" -OutFile $to -UseBasicParsing
                } catch {
                    throw "cannot download $base/$name ($($_.Exception.Message))"
                }
            } else {
                $from = Join-Path $base $name
                if (-not (Test-Path -LiteralPath $from -PathType Leaf)) {
                    throw "cannot find the file $from"
                }
                Copy-Item -LiteralPath $from -Destination $to
            }
        }

        $expected = $null
        foreach ($line in Get-Content -LiteralPath (Join-Path $tmp 'SHA256SUMS')) {
            # `sha256sum` puts `*` before the name of a file that it read as
            # binary.
            $hash, $name = $line.Trim() -split '\s+', 2
            if ($name -and $name.TrimStart('*') -eq $archive) { $expected = $hash }
        }
        if (-not $expected) {
            throw "SHA256SUMS has no line for $archive. The script installed no file."
        }
        $actual = (Get-FileHash -LiteralPath (Join-Path $tmp $archive) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "the SHA-256 of $archive is $actual, and SHA256SUMS records $expected. The script installed no file."
        }

        $extracted = Join-Path $tmp 'extracted'
        Expand-Archive -LiteralPath (Join-Path $tmp $archive) -DestinationPath $extracted
        $new = Join-Path $extracted 'map.exe'
        if (-not (Test-Path -LiteralPath $new -PathType Leaf)) {
            throw "$archive has no ``map.exe``. The script installed no file."
        }

        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        $exe = Join-Path $dir 'map.exe'
        $incoming = "$exe.new"
        $old = "$exe.old"
        # The new binary is in the install directory before a file changes.
        # The two renames that follow are then the only time with no
        # `map.exe`.
        Copy-Item -LiteralPath $new -Destination $incoming -Force
        if (Test-Path -LiteralPath $exe) {
            # Windows refuses to overwrite a program that runs, but it lets a
            # rename through. `map` removes `map.exe.old` at its next start.
            Move-Item -LiteralPath $exe -Destination $old -Force
        }
        Move-Item -LiteralPath $incoming -Destination $exe
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host "Installed $exe"
    $onPath = $env:Path -split ';' | Where-Object { $_.TrimEnd('\') -eq $dir.TrimEnd('\') }
    if (-not $onPath) {
        Write-Host "$dir is not on PATH. To add it, run this command, and then start a new terminal:"
        Write-Host "  [Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';$dir', 'User')"
    }
}

try {
    Install-Map
} catch {
    # `exit` gives a status to the caller of a script file. In `irm | iex`
    # there is no script file: `exit` there closes the terminal of the user.
    # Thus that case gets the error of PowerShell, which also sets a status.
    if (-not $PSCommandPath) { throw "install.ps1: $($_.Exception.Message)" }
    [Console]::Error.WriteLine("install.ps1: $($_.Exception.Message)")
    exit 1
}
