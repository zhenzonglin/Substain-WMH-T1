import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import package_ws1_analysis_project as package


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_project(parent: Path) -> Path:
    root = parent / "Substain"
    for relative in package.REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "envs/offline/environment_archives.sha256":
            continue
        path.write_bytes((relative + "\n").encode("utf-8"))
        if relative.endswith((".sh", "antsRegistration", "mri_synthstrip")):
            path.chmod(0o755)
    for relative in package.REQUIRED_DIRECTORIES:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sentinel.txt").write_text(relative, encoding="utf-8")
    wmh = (root / "envs/offline/wmh-env.tar.gz").read_bytes()
    t1 = (root / "envs/offline/t1-env.tar.gz").read_bytes()
    (root / "envs/offline/environment_archives.sha256").write_text(
        f"{digest(wmh)}  wmh-env.tar.gz\n{digest(t1)}  t1-env.tar.gz\n",
        encoding="utf-8",
    )
    return root


class PackageWorkflowTests(unittest.TestCase):
    def test_dry_run_reports_fixed_paths_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_project(base)

            report = package.build_bundle(root, base, dry_run=True)

            self.assertEqual(report["archive_path"], str(base.resolve() / "Substain_GB.tar.gz"))
            self.assertEqual(report["checksum_path"], str(base.resolve() / "Substain_GB.tar.gz.sha256"))
            self.assertEqual(report["bundle_path"], str(base.resolve() / "Substain_GB_manifest"))
            self.assertEqual(list(base.iterdir()), [root])

    def test_existing_archive_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_project(base)
            archive = base / "Substain_GB.tar.gz"
            archive.write_bytes(b"existing archive")

            with self.assertRaises(package.PackagingError):
                package.build_bundle(root, base)

            self.assertEqual(archive.read_bytes(), b"existing archive")

    def test_collect_excludes_results_inputs_manifests_and_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_project(Path(temporary))
            excluded_files = (
                "BIDS/sub-X/anat/sub-X_T1w.nii.gz",
                "Lesion/sub-X_mask.nii.gz",
                "derivatives/substain_features/sub-X/result.nii.gz",
                "archive/old/source.py",
                "inputs/bids_links/sub-X_T1w.nii.gz",
                "logs/full_run.log",
                "config/participants.tsv",
                "config/metadata.tsv",
                ".git/config",
                ".snakemake/metadata/item",
                "envs/core-venv/bin/python",
                "envs/core-venv.failed-no-ensurepip-20260826-145928/pyvenv.cfg",
                "offline/envs/test/file",
            )
            for relative in excluded_files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("must not be archived", encoding="utf-8")
            included = root / "config/participants.example.tsv"
            included.write_text("participant_id\nEXAMPLE\n", encoding="utf-8")

            entries, excluded = package.collect_entries(root)
            selected = {str(entry["relative_path"]) for entry in entries}

            self.assertIn("config/participants.example.tsv", selected)
            for relative in excluded_files:
                self.assertNotIn(relative, selected)
            self.assertGreater(sum(excluded.values()), 0)

    def test_environment_archives_must_match_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_project(Path(temporary))
            verified = package.verify_environment_archives(root)
            self.assertEqual(set(verified), {"wmh-env.tar.gz", "t1-env.tar.gz"})
            (root / "envs/offline/wmh-env.tar.gz").write_bytes(b"changed")
            with self.assertRaises(package.PackagingError):
                package.verify_environment_archives(root)

    @unittest.skipIf(os.name == "nt", "GNU tar integration is verified under Linux")
    def test_build_bundle_contains_workflow_and_no_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_project(base)
            output = base / "output"
            output.mkdir()
            result = root / "derivatives/substain_features/sub-X/result.nii.gz"
            result.parent.mkdir(parents=True)
            result.write_bytes(b"result")
            (root / "config/participants.tsv").write_text("private", encoding="utf-8")

            report = package.build_bundle(root, output)

            self.assertEqual(report["status"], "pass")
            bundle = Path(str(report["bundle_path"]))
            archive = Path(str(report["archive_path"]))
            checksum = Path(str(report["checksum_path"]))
            self.assertEqual(archive, output.resolve() / "Substain_GB.tar.gz")
            self.assertEqual(bundle, output.resolve() / "Substain_GB_manifest")
            self.assertEqual(checksum.read_text(encoding="utf-8"), f"{digest(archive.read_bytes())}  Substain_GB.tar.gz\n")
            self.assertFalse((bundle / "Substain_GB.tar.gz").exists())
            verification = json.loads((bundle / "VERIFICATION.json").read_text(encoding="utf-8"))
            contents = set((bundle / "CONTENTS.txt").read_text(encoding="utf-8").splitlines())
            self.assertEqual(verification["status"], "pass")
            self.assertIn("Substain/workflow/Snakefile", contents)
            self.assertIn("Substain/envs/offline/wmh-env.tar.gz", contents)
            self.assertNotIn("Substain/config/participants.tsv", contents)
            self.assertFalse(any(member.startswith("Substain/derivatives/") for member in contents))

    @unittest.skipIf(os.name == "nt" or not hasattr(os, "symlink"), "symlink behavior is verified under Linux")
    def test_failed_environment_subtree_is_excluded_from_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_project(base)
            outside = base / "system-python3"
            outside.write_bytes(b"external interpreter")
            failed_env = root / "envs/core-venv.failed-no-ensurepip-20260826-145928"
            failed_bin = failed_env / "bin"
            failed_bin.mkdir(parents=True)
            (failed_env / "pyvenv.cfg").write_text("failed environment", encoding="utf-8")
            os.symlink(str(outside), str(failed_bin / "python3"))
            os.symlink("python3", str(failed_bin / "python"))

            report = package.build_bundle(root, base)

            self.assertEqual(report["status"], "pass")
            bundle = Path(str(report["bundle_path"]))
            contents = (bundle / "CONTENTS.txt").read_text(encoding="utf-8").splitlines()
            self.assertFalse(any(member.startswith("Substain/envs/core-venv.failed-") for member in contents))
            self.assertIn("Substain/envs/offline/t1-env.tar.gz", contents)
            self.assertEqual(report["excluded_rule_counts"]["runtime-env:envs/core-venv.failed-*"], 1)
            self.assertEqual(os.readlink(str(failed_bin / "python")), "python3")
            self.assertEqual(os.readlink(str(failed_bin / "python3")), str(outside))

    @unittest.skipIf(os.name == "nt" or not hasattr(os, "symlink"), "symlink behavior is verified under Linux")
    def test_external_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_project(base)
            outside = base / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            os.symlink(str(outside), str(root / "resources/bad-link"))
            with self.assertRaises(package.PackagingError):
                package.collect_entries(root)


if __name__ == "__main__":
    unittest.main()
