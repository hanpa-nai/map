"""Set the version of every crate in the workspace, then verify it.

A release is a new version number. Without one, `cargo install --git` with a
persistent target directory keeps the old binary: cargo's artifact identity
for a git source does not include the commit, so the same version at a new
commit reads as already built. It prints "Replaced package" and installs the
stale file. Measured on cargo 1.99: 0.0.0 to 0.0.0 kept the old binary, and
0.0.0 to 0.0.1 built the new one.

The version is in the workspace `Cargo.toml`, in the `path` dependency of
every crate on every other crate, and in the README status line. A missed one
is a build error, or a README that names the wrong version. So: rewrite them
all, read them back, and fail loudly unless every count is what it should be.

    python scripts/set-version.py <new-version>

`Cargo.lock` holds the version too, and this script does not touch it. Run
`cargo check --workspace` afterwards and cargo rewrites the workspace entries
there.
"""

import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def write(path: pathlib.Path, text: str) -> None:
    # Bytes, not text mode: a shell script or a hash input must not grow CRs
    # because this ran on Windows.
    path.write_bytes(text.encode("utf-8"))


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    new = sys.argv[1]

    workspace_toml = ROOT / "Cargo.toml"
    text = workspace_toml.read_bytes().decode("utf-8")
    old = tomllib.loads(text)["workspace"]["package"]["version"]
    if old == new:
        print(f"the version is already {new}", file=sys.stderr)
        return 1

    line = re.compile(rf'^version = "{re.escape(old)}"(\r?)$', re.MULTILINE)
    if len(line.findall(text)) != 1:
        print(f"expected one `version = \"{old}\"` line in {workspace_toml}", file=sys.stderr)
        return 1
    write(workspace_toml, line.sub(rf'version = "{new}"\1', text))

    # Every dependency of one workspace crate on another names the version too.
    edited = 0
    dependency = f'version = "{old}", path = '
    for manifest in sorted(ROOT.glob("crates/*/Cargo.toml")):
        text = manifest.read_bytes().decode("utf-8")
        count = text.count(dependency)
        if count:
            write(manifest, text.replace(dependency, f'version = "{new}", path = '))
            edited += count

    readme = ROOT / "README.md"
    text = readme.read_bytes().decode("utf-8")
    status = f"alpha (`v{old}`)"
    if text.count(status) != 1:
        print(f"expected one {status!r} in {readme}", file=sys.stderr)
        return 1
    write(readme, text.replace(status, f"alpha (`v{new}`)"))

    # Read back. Parsed, not grepped: what matters is what cargo will see.
    got = tomllib.loads(workspace_toml.read_bytes().decode("utf-8"))["workspace"]["package"]["version"]
    if got != new:
        print(f"{workspace_toml} still says {got}", file=sys.stderr)
        return 1
    members = {m.parent.name for m in ROOT.glob("crates/*/Cargo.toml")}
    checked = 0
    for manifest in sorted(ROOT.glob("crates/*/Cargo.toml")):
        parsed = tomllib.loads(manifest.read_bytes().decode("utf-8"))
        for name, spec in parsed.get("dependencies", {}).items():
            if name in members:
                if not isinstance(spec, dict) or spec.get("version") != new:
                    print(f"{manifest}: dependency {name} is not at {new}: {spec}", file=sys.stderr)
                    return 1
                checked += 1
    if checked != edited:
        print(f"edited {edited} dependency lines but found {checked} workspace dependencies", file=sys.stderr)
        return 1

    print(f"{old} -> {new}: 1 workspace version, {edited} path dependencies, 1 README line")
    print("next: cargo check --workspace   (it updates Cargo.lock)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
