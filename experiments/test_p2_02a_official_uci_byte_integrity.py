"""Adversarial P2-02A offline tests: synthetic ZIP bytes only, no real data."""
import io
import json
from pathlib import Path
import unittest
import zipfile

from experiments import p2_02a_official_uci_byte_integrity as target


def archive(entries):
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, "w", compression=zipfile.ZIP_DEFLATED) as writer:
        for name, value in entries:
            writer.writestr(name, value)
    return buff.getvalue()


class FrozenSourceByteTests(unittest.TestCase):
    def test_frozen_P1_registry_digest_and_only_three(self):
        protocol = json.loads(target.P1_JSON.read_text(encoding="utf-8"))
        manifest = json.loads(target.P1_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual([x["uci_id"] for x in
                          target.validate_registry(protocol, manifest)],
                         [212, 850, 292])
        protocol["selected_datasets"][0]["zip_sha256"] = "0" * 64
        with self.assertRaises(target.SourceBlocked):
            target.validate_registry(protocol, manifest)

    def test_raw_member_exact_checksum(self):
        value = b"fake ARFF header only, no source rows"
        z = archive([("folder/column_3C_weka.arff", value)])
        spec = {"uci_id": 212, "file_name": "column_3C_weka.arff",
                "file_sha256": target.sha(value)}
        actual = target.member_from_official_zip(z, spec)
        self.assertEqual(actual["selected_member_sha256"], spec["file_sha256"])
        self.assertEqual(actual["actual_data_rows_examined"], 0)
        spec["file_sha256"] = "f" * 64
        with self.assertRaises(target.SourceBlocked):
            target.member_from_official_zip(z, spec)

    def test_nested_Raisin_wrapper_selects_only_pinned_xlsx(self):
        xlsx_bytes = b"synthetic workbook marker, never parsed as XLSX"
        inner = archive([("Raisin_Dataset.xlsx", xlsx_bytes)])
        outer = archive([("Raisin_Dataset.zip", inner), ("README.txt", b"hello")])
        spec = {"uci_id": 850, "file_name": "Raisin_Dataset.xlsx",
                "file_sha256": target.sha(xlsx_bytes)}
        result = target.member_from_official_zip(outer, spec)
        self.assertEqual(result["nested_wrapper"]["nested_zip_sha256_observed"],
                         target.sha(inner))
        self.assertEqual(result["selected_member_sha256"], target.sha(xlsx_bytes))
        self.assertEqual(result["actual_data_rows_examined"], 0)

    def test_rejects_duplicate_source_member(self):
        raw = archive([("a/column_3C_weka.arff", b"1"),
                       ("b/column_3c_weka.ARFF", b"2")])
        spec = {"uci_id": 212, "file_name": "column_3C_weka.arff",
                "file_sha256": target.sha(b"1")}
        with self.assertRaises(target.SourceBlocked):
            target.member_from_official_zip(raw, spec)

    def test_rejects_path_traversal_and_oversized_member(self):
        z = archive([("../secret.csv", b"never allow")])
        with self.assertRaises(target.SourceBlocked):
            target.safe_zip(z)
        with self.assertRaises(target.SourceBlocked):
            target.safe_zip(b"not-a-zip")

    def test_rejects_unapproved_url_and_redirect(self):
        expected = "https://archive.ics.uci.edu/static/public/212/vertebral%2Bcolumn.zip"
        self.assertEqual(target.safe_url(expected, expected=expected), expected)
        for changed in (
            expected.replace("https://", "http://"),
            expected.replace("archive.ics.uci.edu", "malicious.example"),
            expected + "?changed=1",
            "https://archive.ics.uci.edu/static/public/999/not-approved.zip",
        ):
            with self.subTest(url=changed):
                with self.assertRaises(target.SourceBlocked):
                    target.safe_url(changed, expected=expected)


if __name__ == "__main__":
    unittest.main()
