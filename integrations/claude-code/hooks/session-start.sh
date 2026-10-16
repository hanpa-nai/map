#!/bin/sh
#
# The SessionStart hook of the MAP integration for Claude Code.
#
# It runs `map brief`, which prints the index summary for the model. When
# `map` is not on PATH, it looks in the directories where the install scripts,
# cargo, and this integration put the binary, and it adds that directory to
# the PATH of the Bash tool. When no binary exists and the directory has an
# index, it tells the model how to offer the install. It prints nothing when
# the directory has no index.
#
# Claude Code runs this script with the `sh` of Git Bash on Windows. On
# Windows without Git Bash, hooks run in PowerShell, and this script does not
# run.

set -u

# A Windows path such as `C:\Users\me\AppData\Local` in the form that this
# shell accepts. Other systems give the path back unchanged.
posix_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -u "$1"
  else
    printf '%s\n' "$1"
  fi
}

has_map() {
  [ -x "$1/map" ] || [ -f "$1/map.exe" ]
}

# Claude Code gives this directory as a Windows path on Windows.
plugin_bin=
if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then
  plugin_bin=$(posix_path "$CLAUDE_PLUGIN_DATA")/bin
fi
script_bin=
if [ -n "${LOCALAPPDATA:-}" ]; then
  script_bin=$(posix_path "$LOCALAPPDATA")/Programs/map/bin
fi

# The directory that has the binary. Empty when `map` is on PATH already.
dir=
if ! command -v map >/dev/null 2>&1; then
  for candidate in "$plugin_bin" "${HOME:-}/.local/bin" "$script_bin" "${HOME:-}/.cargo/bin"; do
    if [ -n "$candidate" ] && has_map "$candidate"; then
      dir=$candidate
      break
    fi
  done
fi

if [ -n "$dir" ]; then
  PATH=$dir:$PATH
  export PATH
  # Each Bash tool command runs this file first. Thus the model can run `map`
  # with no change to the PATH of the user.
  if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    printf 'export PATH="%s:$PATH"\n' "$dir" >>"$CLAUDE_ENV_FILE"
  fi
fi

if command -v map >/dev/null 2>&1; then
  exec map brief
fi

# No binary. Find the index root the way `map` does: a `.map/config.toml` in
# this directory or in a parent.
root=${CLAUDE_PROJECT_DIR:-$PWD}
config=
while :; do
  if [ -f "$root/.map/config.toml" ]; then
    config=$root/.map/config.toml
    break
  fi
  parent=$(dirname "$root")
  if [ "$parent" = "$root" ]; then
    break
  fi
  root=$parent
done
if [ -z "$config" ]; then
  exit 0
fi

# An index with an embedding dimension or an LLM dimension needs the binary
# with all features. The comment lines of the template name these plugins
# too, so the test skips comments.
variant=default
if grep -v '^[[:space:]]*#' "$config" | grep -Eq 'impl *= *"(distilled|llm)"'; then
  variant=full
fi

install_dir=${plugin_bin:-${HOME:-}/.local/bin}
base=https://github.com/hanpa-nai/map/releases/latest/download
case $(uname -s) in
  MINGW* | MSYS* | CYGWIN*)
    # The model can run this line from the Bash tool or from the PowerShell
    # tool. The two shells give a single-quoted argument to the program
    # unchanged. Windows PowerShell removes double quotes inside an argument,
    # thus the install goes through a script file with plain arguments.
    win_dir=$(cygpath -w "$install_dir")
    win_root=$(cygpath -w "${CLAUDE_PLUGIN_ROOT:-$(dirname "$0")/..}")
    command="powershell -NoProfile -ExecutionPolicy Bypass -File '$win_root\hooks\install.ps1' '$win_dir' '$variant'"
    ;;
  *)
    command="curl -fsSL $base/install.sh | MAP_INSTALL_DIR='$install_dir' MAP_INSTALL_VARIANT='$variant' sh"
    ;;
esac

cat <<MESSAGE
This repository has a MAP index in \`.map/\`, and the \`map\` binary is not installed.
Ask the user for the install. Do not run the install without their consent.
The install command downloads the newest release of MAP (one archive, less than 5 MB), checks its SHA-256, and puts \`map\` in \`$install_dir\`. It changes no other file and no shell profile:
  $command
After the install, run \`$install_dir/map brief\` and follow its output. In this session, use \`$install_dir/map\` in place of \`map\`. From the next session, \`map\` is on the PATH of the Bash tool.
If the user declines, use your usual search tools. The install instructions are at https://github.com/hanpa-nai/map#install.
MESSAGE
