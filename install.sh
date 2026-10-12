#!/bin/sh
#
# Install the `map` binary of the newest release. Rust is not necessary.
#
#     curl -fsSL https://github.com/hanpa-nai/map/releases/latest/download/install.sh | sh
#
# The script downloads one archive and the file `SHA256SUMS`. It stops if the
# SHA-256 of the archive is not the recorded one. Then it puts `map` in the
# install directory. It does not change `PATH` or a shell profile.
#
# Environment variables:
#
#   MAP_INSTALL_DIR      The install directory. Default: ~/.local/bin
#   MAP_INSTALL_VARIANT  `full` gets the binary with all features.
#                        Default: `default`, the default binary.
#   MAP_INSTALL_BASE     The location of the release files: the start of a
#                        URL, or a local directory. Default: the newest
#                        release on GitHub.
#   MAP_INSTALL_TARGET   The target of the binary, such as
#                        x86_64-unknown-linux-musl. Default: the target of
#                        this machine.
#
# This is POSIX `sh`, not bash: the `sh` of Debian and of Alpine is not bash.

set -eu

fail() {
  printf 'install.sh: %s\n' "$1" >&2
  exit 1
}

has() {
  command -v "$1" >/dev/null 2>&1
}

# The target of this machine. The Linux binary is static: it does not use the
# C library of the distribution. Thus the script does not examine which C
# library the machine has.
find_target() {
  os=$(uname -s)
  cpu=$(uname -m)
  case "$os" in
    Linux) system=unknown-linux-musl ;;
    Darwin) system=apple-darwin ;;
    MINGW* | MSYS* | CYGWIN*)
      fail "this script is not for Windows. In PowerShell, run: irm $default_base/install.ps1 | iex" ;;
    *) fail "no binary is available for the operating system $os. Build MAP with cargo." ;;
  esac
  case "$cpu" in
    x86_64 | amd64) cpu=x86_64 ;;
    aarch64 | arm64) cpu=aarch64 ;;
    *) fail "no binary is available for the CPU $cpu. Build MAP with cargo." ;;
  esac
  printf '%s-%s\n' "$cpu" "$system"
}

# fetch NAME FILE: get the release file NAME into FILE.
fetch() {
  case "$base" in
    *://*)
      if has curl; then
        curl -fsSL "$base/$1" -o "$2" || fail "cannot download $base/$1"
      else
        wget -q -O "$2" "$base/$1" || fail "cannot download $base/$1"
      fi
      ;;
    *)
      [ -f "$base/$1" ] || fail "cannot find the file $base/$1"
      cp "$base/$1" "$2"
      ;;
  esac
}

# The SHA-256 of a file, in lower-case hex. macOS has `shasum` and no
# `sha256sum`.
sha256_of() {
  if has sha256sum; then
    sha256sum <"$1" | cut -d ' ' -f 1
  else
    shasum -a 256 <"$1" | cut -d ' ' -f 1
  fi
}

# The hash that `SHA256SUMS` records for a file name. Empty if it has none.
recorded_sha256() {
  cr=$(printf '\r')
  while read -r hash name || [ -n "$hash" ]; do
    # `sha256sum` puts `*` before the name of a file that it read as binary,
    # and a file from Windows can have a carriage return at each line end.
    name=${name%"$cr"}
    name=${name#\*}
    if [ "$name" = "$2" ]; then
      printf '%s\n' "$hash"
      return
    fi
  done <"$1"
}

main() {
  default_base=https://github.com/hanpa-nai/map/releases/latest/download
  base=${MAP_INSTALL_BASE:-$default_base}
  base=${base%/}
  variant=${MAP_INSTALL_VARIANT:-default}
  target=${MAP_INSTALL_TARGET:-}

  case "$variant" in
    default) prefix=map ;;
    full) prefix=map-full ;;
    *) fail "MAP_INSTALL_VARIANT is \`$variant\`. Use \`default\` or \`full\`." ;;
  esac

  if [ -z "$target" ]; then
    target=$(find_target)
  fi
  case "$target" in
    *windows*) fail "the binary for $target is a zip. In PowerShell, run: irm $default_base/install.ps1 | iex" ;;
  esac

  if [ -n "${MAP_INSTALL_DIR:-}" ]; then
    dir=$MAP_INSTALL_DIR
  elif [ -n "${HOME:-}" ]; then
    dir=$HOME/.local/bin
  else
    fail "HOME is not set. Set MAP_INSTALL_DIR to the install directory."
  fi

  # The script looks for each necessary program before the first download.
  has tar || fail "\`tar\` is necessary to extract the archive. Install it, then run this script again."
  has sha256sum || has shasum ||
    fail "\`sha256sum\` or \`shasum\` is necessary to examine the archive. Install one, then run this script again."
  case "$base" in
    *://*)
      has curl || has wget ||
        fail "\`curl\` or \`wget\` is necessary to download the archive. Install one, then run this script again."
      ;;
  esac

  tmp=$(mktemp -d 2>/dev/null || mktemp -d -t map-install)
  # `EXIT` covers an error and the end of the script. A shell does not run
  # the `EXIT` commands for each signal. Thus a signal becomes an exit.
  trap 'rm -rf "$tmp"' EXIT
  trap 'exit 1' HUP INT TERM

  archive=$prefix-$target.tar.gz
  case "$base" in
    *://*) printf 'Downloading %s\n' "$base/$archive" ;;
    *) printf 'Copying %s\n' "$base/$archive" ;;
  esac
  fetch "$archive" "$tmp/$archive"
  fetch SHA256SUMS "$tmp/SHA256SUMS"

  expected=$(recorded_sha256 "$tmp/SHA256SUMS" "$archive")
  [ -n "$expected" ] || fail "SHA256SUMS has no line for $archive. The script installed no file."
  actual=$(sha256_of "$tmp/$archive")
  [ "$actual" = "$expected" ] ||
    fail "the SHA-256 of $archive is $actual, and SHA256SUMS records $expected. The script installed no file."

  mkdir "$tmp/extracted"
  tar -xzf "$tmp/$archive" -C "$tmp/extracted" ||
    fail "cannot extract $archive. The script installed no file."
  [ -f "$tmp/extracted/map" ] || fail "$archive has no \`map\` binary. The script installed no file."

  mkdir -p "$dir" || fail "cannot make the directory $dir"
  # A copy into the install directory, then a rename. The rename replaces a
  # `map` in one step, and Linux refuses a write to a program that runs.
  cp "$tmp/extracted/map" "$dir/map.new" || fail "cannot write in $dir"
  chmod 755 "$dir/map.new"
  mv -f "$dir/map.new" "$dir/map" || fail "cannot replace $dir/map"

  printf 'Installed %s\n' "$dir/map"
  case ":$PATH:" in
    *":$dir:"*) ;;
    *)
      printf '%s is not on PATH. To add it, put this line in the profile of your shell:\n' "$dir"
      printf '  export PATH="%s:$PATH"\n' "$dir"
      ;;
  esac
}

# The body of the script is functions, and this line is the only call. Thus a
# download that stops before the end of the file runs no command.
main "$@"
