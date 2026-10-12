import tempfile
import unittest
from pathlib import Path
from docs.validate_stopping_reproduction import validate


class ReproductionTests(unittest.TestCase):
    def test_csv_line_endings_and_changed_scientific_value(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';output=root/'output'
            source.mkdir();output.mkdir()
            (source/'individual.csv').write_bytes(b'iteration,accuracy\n1000,0.75\n')
            (output/'individual.csv').write_bytes(b'iteration,accuracy\r\n1000,0.75\r\n')
            self.assertEqual(validate(output,source,required=['individual.csv']),['individual.csv'])
            (output/'individual.csv').write_text('iteration,accuracy\n1000,0.76\n')
            with self.assertRaises(ValueError):validate(output,source)

    def test_missing_outputs_or_references_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(ValueError):validate(root,root,required=['individual.csv'])
            (root/'individual.csv').write_text('iteration\n1000\n')
            absent=root/'absent';absent.mkdir()
            with self.assertRaises(ValueError):validate(root,absent,required=['individual.csv'])

    def test_changed_figure_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';output=root/'output'
            source.mkdir();output.mkdir()
            (source/'figure.png').write_bytes(b'original')
            (output/'figure.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):validate(output,source)
