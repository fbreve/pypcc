"""P2-06 no-data archive integrity tests; never run the PCC here."""
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from experiments import p2_06_synthetic_parity_exact_artifacts as target


class P206ArchiveTests(unittest.TestCase):
    def test_binary_header_elf_x86_64_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "binary.so"
            valid = bytearray(64)
            valid[:4] = b"\x7fELF"
            valid[4] = 2
            valid[5] = 1
            valid[18:20] = struct.pack("<H", 62)
            path.write_bytes(valid)
            self.assertEqual(target.native_header(path)["machine"], 62)
            valid[18:20] = struct.pack("<H", 3)
            path.write_bytes(valid)
            with self.assertRaises(target.ParityBlocked):
                target.native_header(path)

    def test_binary_header_pe_x86_64_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "binary.pyd"
            valid = bytearray(128)
            valid[:2] = b"MZ"
            valid[0x3c:0x40] = struct.pack("<I", 64)
            valid[64:68] = b"PE\x00\x00"
            valid[68:70] = struct.pack("<H", 0x8664)
            path.write_bytes(valid)
            self.assertEqual(target.native_header(path)["machine"], 0x8664)
            valid[68:70] = struct.pack("<H", 0x14c)
            path.write_bytes(valid)
            with self.assertRaises(target.ParityBlocked):
                target.native_header(path)

    def test_pinned_tests_not_committed_to_this_branch(self):
        self.assertEqual(len(target.TEST_BLOBS), 3)
        self.assertEqual(set(target.MODULES), {
            "production_pcc_step", "isolated_pcc_step_live_research",
        })
        self.assertEqual(target.P2_01C_HEAD,
                         "fa71adb7c4defa54de9d233de4b0bc84544d1c24")


if __name__ == "__main__":
    unittest.main()
