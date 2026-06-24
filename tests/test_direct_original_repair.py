from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from svbr.experiments.direct_original_repair import CsvSink, completed_case_ids, shard_accepts


class DirectOriginalRepairTest(unittest.TestCase):
    def test_csv_sink_resume_appends_without_duplicate_header(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "runs.csv"
            fields = ["case_id", "value"]
            with CsvSink(path, fields) as sink:
                sink.writerow({"case_id": "case-a", "value": "1"})
            with CsvSink(path, fields, append=True) as sink:
                sink.writerow({"case_id": "case-b", "value": "2"})

            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(["case-a", "case-b"], [row["case_id"] for row in rows])
            self.assertEqual({"case-a", "case-b"}, completed_case_ids(path))

    def test_shards_partition_case_ids_exactly_once(self):
        case_ids = [f"case-{index}" for index in range(100)]
        shard_count = 4
        for case_id in case_ids:
            accepted = [index for index in range(shard_count) if shard_accepts(case_id, index, shard_count)]
            self.assertEqual(1, len(accepted), case_id)


if __name__ == "__main__":
    unittest.main()
