"""Pure synthetic log/metadata parser tests. NEVER compile or invoke PCC."""
from pathlib import Path
import tempfile
import unittest

from experiments import p2_01f_prospective_native_build_ledger as candidate


class ProspectiveBuildLedgerTests(unittest.TestCase):
    def test_linux_original_compiler_and_link_commands_are_required(self):
        log = ("building 'pcc_step' extension\n"
               "gcc -DNDEBUG -O3 -fPIC -c pcc_step.c -o build/pcc_step.o -O3\n"
               "gcc -shared build/pcc_step.o -o pcc_step.cpython-312-x86_64-linux-gnu.so\n")
        result = candidate.find_compile_and_link(log, "Linux", "pcc_step")
        self.assertEqual(len(result["compile_commands"]), 1)
        self.assertEqual(len(result["link_commands"]), 1)
        for missing in (log.splitlines()[1], log.splitlines()[2], ""):
            with self.assertRaises(candidate.NativeBuildBlocked):
                candidate.find_compile_and_link(missing, "Linux", "pcc_step")

    def test_windows_MSVC_original_command_capture(self):
        log = (
            '"C:\\Program Files\\Visual Studio\\Tools\\cl.exe" /c /O2 '
            '/TcD:\\build\\pcc_step_live_research.c '
            '/FoD:\\build\\pcc_step_live_research.obj\n'
            '"C:\\Program Files\\Visual Studio\\Tools\\link.exe" /DLL '
            '/OUT:D:\\build\\pcc_step_live_research.cp312-win_amd64.pyd '
            'D:\\build\\pcc_step_live_research.obj\n'
        )
        result = candidate.find_compile_and_link(
            log, "Windows", "pcc_step_live_research")
        self.assertEqual(len(result["compile_commands"]), 1)
        self.assertEqual(len(result["link_commands"]), 1)
        with self.assertRaises(candidate.NativeBuildBlocked):
            candidate.find_compile_and_link(
                log.replace("/OUT:", "/IMPLIB:"), "Windows",
                "pcc_step_live_research")

    def test_real_file_bytes_are_sha_hashed_and_empty_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"compiler.exe"
            p.write_bytes(b"synthetic compiler marker only")
            result = candidate.binary_file(p)
            self.assertEqual(len(result["sha256"]), 64)
            self.assertEqual(result["bytes"], p.stat().st_size)
            p.write_bytes(b"modified marker")
            self.assertNotEqual(candidate.binary_file(p)["sha256"], result["sha256"])
            p.write_bytes(b"")
            with self.assertRaises(candidate.NativeBuildBlocked):
                candidate.binary_file(p)

    def test_missing_production_link_blocks_and_P1_is_exact(self):
        self.assertEqual(candidate.P1_SHA,
                         "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166")
        with self.assertRaises(candidate.NativeBuildBlocked):
            candidate.find_compile_and_link(
                'building "pcc_step" extension only, no actual gcc command',
                "Linux", "pcc_step")

    def test_no_claim_of_unchanged_compiler_without_command(self):
        with self.assertRaises(candidate.NativeBuildBlocked):
            candidate.find_compile_and_link("", "Windows",
                                            "pcc_step_live_research")


if __name__ == "__main__":
    unittest.main()
