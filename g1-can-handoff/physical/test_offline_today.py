"""Report failure visibility and separation from physical readiness."""

from pathlib import Path
import tempfile
import unittest

from offline.today import COVERAGE, MODULES, dashboard


class TodayReportTests(unittest.TestCase):
    def test_catalog_has_no_duplicate_suites(self):
        self.assertEqual(len(MODULES), len(set(MODULES)))
        self.assertTrue(
            all(
                (Path(__file__).parent / (module + ".py")).is_file()
                for module in MODULES
            )
        )

    def test_failure_and_custom_report_name_remain_visible(self):
        report = dict(
            successful=False,
            tests_run=1,
            evidence_verified=143,
            seconds=1.0,
            modules=[
                dict(
                    module=MODULES[0],
                    successful=False,
                    tests_run=1,
                    cases=[
                        dict(test="failed<case>", status="failed", detail="bad<script>")
                    ],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "index.html"
            dashboard(report, destination, "results.json")
            text = destination.read_text()
        self.assertIn("FAIL · 1 tests", text)
        self.assertIn("bad&lt;script&gt;", text)
        self.assertIn('href="results.json"', text)
        self.assertIn("does not mean every captured situation permits a pickup", text)
        self.assertNotIn("bad<script>", text)
        self.assertIn("NOT RUN", text)

    def test_old_animations_are_not_used_by_a_new_run(self):
        report = dict(
            successful=True,
            tests_run=0,
            evidence_verified=143,
            seconds=1.0,
            modules=[],
            simulation_artifact_dir="simulations/new-run",
        )
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "simulations/old-run/case"
            old.mkdir(parents=True)
            (old / "result.json").write_text("invalid old result must not be parsed")
            dashboard(report, Path(directory) / "index.html")


if __name__ == "__main__":
    unittest.main()
