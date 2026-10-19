import json
import shutil
import tempfile
import unittest
from pathlib import Path

from datavault.repo import Repository

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


class RepoTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.repo = Repository.init(self.root)

    def write(self, name: str, content: str | bytes) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
        return path

    def commit_file(self, name: str, content: str | bytes, message: str = "c", **kw):
        path = self.write(name, content)
        self.repo.add([path])
        return self.repo.commit(message, **kw)

    def copy_example(self, name: str) -> Path:
        target = self.root / name
        shutil.copy(EXAMPLES / name, target)
        return target


LABELED = (
    "great product works well|positive\nlove it so much|positive\nfantastic quality|positive\n"
    "really happy with it|positive\nexcellent value|positive\nwould buy again|positive\n"
    "terrible broke fast|negative\nawful support|negative\nwaste of money|negative\n"
    "very disappointed|negative\npoor build quality|negative\nnever again|negative\n"
)


def load(path: Path):
    return json.loads(path.read_text())
