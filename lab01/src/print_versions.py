import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "results" / "versions.txt"


def package_names():
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return [l.split("==")[0].strip() for l in lines if l.strip() and not l.startswith("#")]


def main():
    rows = [f"Python {sys.version.split()[0]}", f"Platform {platform.platform()}"]
    for name in package_names():
        try:
            rows.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            rows.append(f"{name}: NOT INSTALLED")
    text = "\n".join(rows)
    print(text)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text + "\n")


if __name__ == "__main__":
    main()
