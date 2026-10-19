import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dv(workdir: Path, *args: str, check: bool = True) -> None:
    print("$ datavault " + " ".join(args))
    done = subprocess.run([sys.executable, "-m", "datavault", *args], cwd=workdir, text=True)
    if check and done.returncode:
        sys.exit(done.returncode)
    print()


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="datavault-demo-"))
    for name in ("reviews.csv", "quality_rules.json", "experiment.json"):
        shutil.copy(HERE / name, work / name)
    rows = (work / "reviews.csv").read_text().splitlines(keepends=True)

    dv(work, "init")
    dv(work, "add", "reviews.csv")
    dv(work, "commit", "-m", "initial reviews", "--checks", "quality_rules.json")
    dv(work, "tag", "v1")

    degraded = [rows[0]] + [r.replace(",positive,", ",negative,", 1) if i % 4 == 0 else r for i, r in enumerate(rows[1:])]
    degraded += [f"{n},,positive,\n" for n in range(61, 67)]
    (work / "reviews.csv").write_text("".join(degraded))
    dv(work, "add", "reviews.csv")
    dv(work, "commit", "-m", "relabel batch and add sparse rows", "--checks", "quality_rules.json")

    dv(work, "log")
    dv(work, "diff", "v1", "HEAD")
    dv(work, "verify", "--all")
    dv(work, "experiment", "run", "v1", "--config", "experiment.json")
    dv(work, "experiment", "run", "HEAD", "--config", "experiment.json")
    dv(work, "experiment", "list")

    ids = subprocess.run([sys.executable, "-m", "datavault", "experiment", "list", "--json"], cwd=work,
                         capture_output=True, text=True, check=True).stdout
    import json
    first, second = (r["id"][:12] for r in json.loads(ids))
    dv(work, "experiment", "compare", first, second)
    dv(work, "experiment", "reproduce", first)

    if os.environ.get("DATAVAULT_AI_PROVIDER"):
        dv(work, "ai", "dataset", "v1", "HEAD", check=False)
    else:
        print("AI step skipped: set DATAVAULT_AI_PROVIDER and DATAVAULT_AI_MODEL to enable it.\n")

    print(f"repository: {work}\nview it: datavault -C {work} serve")
    if not keep:
        print("(the directory is left in place so the dashboard can open it; delete it when done)")


if __name__ == "__main__":
    main()
