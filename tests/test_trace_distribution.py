"""Release checks must detect missing inputs and stale/private wheel contents."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import check_trace_distribution as checker


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        names = ["README.md", "LICENSE", "NOTICE.md", "MANIFEST.in", "pyproject.toml",
                 "native/reference_cache.cpp", "tests/smoke_trace_install.py",
                 "tests/check_trace_distribution.py", "hbserve/__init__.py",
                 "hbserve/traces/SOURCES.json", "hbserve/traces/LICENSE.HBFSim",
                 "configs/system.cfg", "docs/trace-replay.md"]
        self.files = {name: name.encode() for name in names}
        for name, body in self.files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        self.patcher = patch.object(checker, "ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def archives(self, *, omit=None, stale=None, extra=None):
        source = self.root / "source.tar.gz"
        wheel = self.root / "release.whl"
        with tarfile.open(source, "w:gz") as archive:
            for name, body in self.files.items():
                if name == omit:
                    continue
                member = tarfile.TarInfo("hbserve/" + name)
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, body in self.files.items():
                if name.startswith(("hbserve/", "hbfsim_client/")):
                    archive.writestr(name, b"old" if name == stale else body)
            if extra:
                archive.writestr(extra, b"untracked")
        return source, wheel

    def test_complete_distribution_passes(self):
        self.assertEqual(checker.check(*self.archives())["status"], "PASS_DISTRIBUTION_CONTENTS")

    def test_source_config_and_replay_docs_are_required(self):
        for name in ("configs/system.cfg", "docs/trace-replay.md"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing"):
                checker.check(*self.archives(omit=name))

    def test_stale_installed_runtime_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale wheel"):
            checker.check(*self.archives(stale="hbserve/__init__.py"))

    def test_untracked_experimental_module_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "untracked runtime"):
            checker.check(*self.archives(extra="hbserve/experimental.py"))


if __name__ == "__main__":
    unittest.main()
