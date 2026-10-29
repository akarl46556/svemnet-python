"""Check release versions and archive boundaries without importing svemnet."""
from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = {
    ".git", ".codex", ".claude", ".venv", "__pycache__", ".pytest_cache",
    "project_review", "diagnostics", "development", "tmp", "output",
    "AGENTS.md", "CLAUDE.md",
}
SDIST_ROOTS = {
    "src", "tests", "examples", "validation", "tools", "README.md",
    "CHANGELOG.md", "LICENSE", "CITATION.cff", "pyproject.toml",
    "RELEASING.md", "PKG-INFO", ".gitignore",
}


def source_version() -> str:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    module = (ROOT / "src/svemnet/__init__.py").read_text(encoding="utf-8")
    versions = [
        re.findall(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE),
        re.findall(r'^__version__\s*=\s*"([^"]+)"', module, re.MULTILINE),
    ]
    if any(len(v) != 1 for v in versions) or versions[0] != versions[1]:
        raise ValueError(f"source version mismatch: {versions}")
    return versions[0][0]


def inspect_archive(path: Path, version: str) -> int:
    wheel = path.suffix == ".whl"
    if wheel:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError(f"{path.name}: expected one METADATA")
            info = archive.read(metadata[0]).decode("utf-8")
    else:
        with tarfile.open(path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            names = [member.name for member in members]
            metadata = [member for member in members if PurePosixPath(member.name).parts == (f"svemnet-{version}", "PKG-INFO")]
            if len(metadata) != 1:
                raise ValueError(f"{path.name}: expected top-level PKG-INFO")
            stream = archive.extractfile(metadata[0])
            assert stream is not None
            info = stream.read().decode("utf-8")
    parsed = Parser().parsestr(info)
    if parsed["Name"] != "svemnet" or parsed["Version"] != version:
        raise ValueError(f"{path.name}: metadata version/name mismatch")
    for name in names:
        parts = PurePosixPath(name).parts
        if not parts or ".." in parts or PurePosixPath(name).is_absolute():
            raise ValueError(f"{path.name}: unsafe member {name}")
        if any(part in PRIVATE for part in parts) or name.endswith((".pyc", ".pyo")):
            raise ValueError(f"{path.name}: private/generated member {name}")
        if wheel:
            if parts[0] not in {"svemnet", f"svemnet-{version}.dist-info"}:
                raise ValueError(f"{path.name}: unexpected wheel member {name}")
        elif len(parts) < 2 or parts[0] != f"svemnet-{version}" or parts[1] not in SDIST_ROOTS:
            raise ValueError(f"{path.name}: unexpected sdist member {name}")
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, help="directory containing the built wheel and sdist")
    parser.add_argument("--tag", help="optional release tag, exactly v<source version>")
    args = parser.parse_args()
    version = source_version()
    if args.tag is not None and args.tag != f"v{version}":
        raise ValueError(f"tag {args.tag!r} does not match source v{version}")
    archives = sorted([*args.dist.glob("*.whl"), *args.dist.glob("*.tar.gz")])
    if len(archives) != 2 or not any(p.suffix == ".whl" for p in archives):
        raise ValueError("check a clean distribution directory containing one wheel and one sdist")
    for path in archives:
        print(f"{path.name}: {inspect_archive(path, version)} public files; version {version}; boundary OK")


if __name__ == "__main__":
    main()

