from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_train_associations_v002.py"
SPEC = importlib.util.spec_from_file_location("train_association_v002", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import contract failure
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class TrainAssociationModeTests(unittest.TestCase):
    def test_quick_contract_is_isolated_and_nonrelease(self) -> None:
        config = analysis.parse_args(["--quick"])
        self.assertEqual(config.mode, "quick")
        self.assertGreaterEqual(config.threads, 1)
        self.assertFalse(config.full_input_sha256)
        self.assertFalse(config.checkpoint_eligible)
        self.assertFalse(config.render_chart)
        self.assertNotEqual(config.output_dir, analysis.RELEASE_OUTPUT_DIR)
        self.assertNotEqual(config.report_path, analysis.RELEASE_REPORT_PATH)
        self.assertEqual(config.verification_level, "manifest_membership_and_size")

    def test_quick_optional_controls(self) -> None:
        config = analysis.parse_args(["--quick", "--threads", "3", "--with-chart"])
        self.assertEqual(config.threads, 3)
        self.assertTrue(config.render_chart)

    def test_release_contract_is_full_and_single_threaded(self) -> None:
        config = analysis.parse_args(["--release"])
        self.assertEqual(config.mode, "release")
        self.assertEqual(config.threads, 1)
        self.assertTrue(config.full_input_sha256)
        self.assertTrue(config.checkpoint_eligible)
        self.assertTrue(config.render_chart)
        self.assertEqual(config.output_dir, analysis.RELEASE_OUTPUT_DIR)
        self.assertEqual(config.report_path, analysis.RELEASE_REPORT_PATH)
        self.assertEqual(config.verification_level, "full_sha256")

    def test_mode_is_required_and_mutually_exclusive(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                analysis.parse_args([])
            with self.assertRaises(SystemExit):
                analysis.parse_args(["--quick", "--release"])

    def test_release_rejects_thread_override(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                analysis.parse_args(["--release", "--threads", "8"])

    def test_accelerator_manifest_does_not_claim_gpu(self) -> None:
        probe = analysis.accelerator_probe()
        self.assertEqual(probe["selected_backend"], "duckdb_cpu")
        self.assertFalse(probe["accelerator_used"])
        self.assertFalse(probe["gpu_used"])


if __name__ == "__main__":
    unittest.main()
