"""P2-01D unit gates: synthetic ELF/PE byte fixtures, no native imports or PCC."""
import struct
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from experiments import p2_01d_archived_binary_wheel_forensics as audit


def elf_fixture(needed="libc.so.6"):
    data = bytearray(512)
    data[:4] = bytes((127,)) + b"ELF"
    data[4] = 2
    data[5] = 1
    struct.pack_into("<H", data, 18, 62)
    struct.pack_into("<Q", data, 32, 64)
    struct.pack_into("<H", data, 54, 56)
    struct.pack_into("<H", data, 56, 2)
    # PT_LOAD virtual 0x1000 maps to file offset zero.
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 4, 0, 0x1000, 0, 512, 512, 0x1000)
    struct.pack_into("<IIQQQQQQ", data, 120, 2, 4, 256, 0x1100, 0, 64, 64, 8)
    string_bytes = needed.encode("ascii") + bytes((0,))
    data[384:384+len(string_bytes)] = string_bytes
    for pos, tag, value in ((256, 5, 0x1180), (272, 10, len(string_bytes)),
                            (288, 1, 0), (304, 0, 0)):
        struct.pack_into("<qQ", data, pos, tag, value)
    return data


def pe_fixture(dll="python312.dll"):
    data = bytearray(1024)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3c, 0x80)
    data[0x80:0x84] = b"PE" + bytes((0, 0))
    struct.pack_into("<H", data, 0x84, 0x8664)  # AMD64
    struct.pack_into("<H", data, 0x86, 1)       # one section
    struct.pack_into("<H", data, 0x94, 240)     # optional size
    opt = 0x98
    struct.pack_into("<H", data, opt, 0x20b)
    struct.pack_into("<II", data, opt+120, 0x1000, 40)
    section = opt + 240
    data[section:section+6] = b".idata"
    struct.pack_into("<IIII", data, section+8, 512, 0x1000, 512, 0x200)
    # Import descriptor at raw 0x200, referenced RVA 0x1000.
    struct.pack_into("<IIIII", data, 0x200, 0, 0, 0, 0x1050, 0)
    name_bytes = dll.encode("ascii") + bytes((0,))
    data[0x250:0x250+len(name_bytes)] = name_bytes
    return data


class NativeForensicsUnitTests(unittest.TestCase):
    def test_elf_needed_and_corrupted_architecture(self):
        data = elf_fixture()
        self.assertEqual(audit.inspect_elf(data)["DT_NEEDED"], ["libc.so.6"])
        struct.pack_into("<H", data, 18, 3)
        with self.assertRaises(audit.AuditBlocked):
            audit.inspect_elf(data)

    def test_pe_imports_and_corrupted_architecture(self):
        data = pe_fixture()
        self.assertEqual(audit.inspect_pe(data)["imported_DLL_names"], ["python312.dll"])
        struct.pack_into("<H", data, 0x84, 0x14c)
        with self.assertRaises(audit.AuditBlocked):
            audit.inspect_pe(data)

    def test_wheel_name_tags_and_internal_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel = root / "example_dist-1.2.3-py3-none-any.whl"
            with ZipFile(wheel, "w") as z:
                z.writestr("example_dist-1.2.3.dist-info/METADATA",
                           "Metadata-Version: 2.1\nName: example-dist\nVersion: 1.2.3\n")
            result = audit.candidate_wheel_inventory(root, {"example-dist": "1.2.3"})
            self.assertEqual(result["example-dist"]["bytes"], wheel.stat().st_size)
            with self.assertRaises(audit.AuditBlocked):
                audit.candidate_wheel_inventory(root, {"example-dist": "1.2.4"})

    def test_wrong_wheel_metadata_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel = root / "example_dist-1.2.3-py3-none-any.whl"
            with ZipFile(wheel, "w") as z:
                z.writestr("example_dist-1.2.3.dist-info/METADATA",
                           "Metadata-Version: 2.1\nName: wrong-name\nVersion: 1.2.3\n")
            with self.assertRaises(audit.AuditBlocked):
                audit.candidate_wheel_inventory(root, {"example-dist": "1.2.3"})


if __name__ == "__main__":
    unittest.main()
