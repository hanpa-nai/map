<#
.SYNOPSIS
  Clone the pinned example corpora as siblings of the map repo.

.DESCRIPTION
  Corpora live NEXT TO this repository, not inside it:

      parent/
      |- map/       <- this repo
      |- ripgrep/   <- corpus
      \- flask/     <- corpus

  Pins are exact commits. This is load-bearing: the qrels.jsonl of a corpus
  in eval/corpora judges specific line ranges, and eval/baseline-*.json
  freezes the scores those
  judgments produce, so both rot silently against a moving corpus. Do not
  change a pin without re-authoring the judgments and re-freezing the
  baselines.

  The stage fingerprint ledger does NOT depend on these pins - it runs over an
  inline fixture in map-index. See crates/map-index/src/ledger.rs.
#>
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$repoRoot  = Split-Path -Parent $PSScriptRoot
$parent    = Split-Path -Parent $repoRoot

$corpora = @(
    @{ Name = 'ripgrep'; Url = 'https://github.com/BurntSushi/ripgrep.git'; Pin = '8372866810a1f2a647d11d7780984d4402a5c1e9' }
    @{ Name = 'flask';   Url = 'https://github.com/pallets/flask.git';      Pin = '36e4a824f340fdee7ed50937ba8e7f6bc7d17f81' }
)

foreach ($c in $corpora) {
    $dest = Join-Path $parent $c.Name

    if (Test-Path $dest) {
        $have = (git -C $dest rev-parse HEAD).Trim()
        if ($have -eq $c.Pin) {
            Write-Host "ok    $($c.Name)  $($c.Pin.Substring(0,12))"
            continue
        }
        if (-not $Force) {
            Write-Warning "$($c.Name) is at $($have.Substring(0,12)), expected $($c.Pin.Substring(0,12)). Re-run with -Force to reset."
            continue
        }
        Write-Host "reset $($c.Name) -> $($c.Pin.Substring(0,12))"
        git -C $dest fetch --depth 1 origin $c.Pin
        git -C $dest checkout --detach FETCH_HEAD
        continue
    }

    Write-Host "fetch $($c.Name)  $($c.Pin.Substring(0,12))"
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    git -C $dest init -q
    git -C $dest remote add origin $c.Url
    # Fetch the single pinned commit rather than history we will never read.
    git -C $dest fetch --depth 1 origin $c.Pin
    git -C $dest checkout --detach FETCH_HEAD

    $got = (git -C $dest rev-parse HEAD).Trim()
    if ($got -ne $c.Pin) { throw "$($c.Name): expected $($c.Pin), got $got" }
}

Write-Host ""
Write-Host "corpora root: $parent"
