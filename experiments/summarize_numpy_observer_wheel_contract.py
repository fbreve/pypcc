"""Fail closed unless all four independent installed-wheel API reports pass."""
import argparse
import json
from pathlib import Path

OSES=("Linux","Windows")
PYTHONS=("3.10","3.12")
REQUIRED=(
    "installed_module_origins","wheel_contains_numpy_module",
    "wheel_excludes_experiments","public_fit_has_no_observer",
    "private_trailing_observer_default_none","high_level_rejects_observer",
    "six_final_arrays_exact","native_stop_info_exact","complete_rng_exact",
    "next16_exact","exact_actual_schedule","genuine_terminal_info",
    "terminal_state_rng_exact",
)


def collect(paths):
    expected={(os,version) for os in OSES for version in PYTHONS}
    records={}
    for path in paths:
        report=json.loads(Path(path).read_text(encoding="utf-8"))
        platform=report.get("platform")
        python=".".join(str(report.get("python","")).split(".")[:2])
        key=(platform,python)
        if key not in expected:
            raise AssertionError(f"Unexpected platform/Python: {key}")
        if key in records:
            raise AssertionError(f"Duplicate platform/Python: {key}")
        checks=report.get("checks",{})
        if (report.get("passed") is not True
                or any(checks.get(k) is not True for k in REQUIRED)):
            raise AssertionError(f"Failed or incomplete wheel contract: {key}")
        if report.get("package_name","").replace("-","_").lower()!="pcc_step":
            raise AssertionError(f"Wrong installed wheel distribution: {key}")
        if report.get("snapshot_iterations")!=[0,1,21]:
            raise AssertionError(f"Wrong installed wheel timeline: {key}")
        records[key]=report
    if set(records)!=expected:
        raise AssertionError(f"Missing installed-wheel reports: {sorted(expected-set(records))}")
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files",nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    reports=collect(args.files)
    lines=[
        "# NumPy observer installed-wheel phase 5: 4/4 strict contract passes",
        "",
        "Built from isolated sdist to wheel, installed and tested outside the checkout with Python -I.",
        "",
        "| Platform | Python | Package version | Native wheel imports | Exact parity |",
        "|---|---|---|---|---|",
    ]
    for key,report in sorted(reports.items()):
        lines.append(
            f"| {key[0]} | {report['python']} | {report['package_version']} | "
            "PASS | PASS |")
    lines.extend([
        "",
        "13/13 contract assertions per job pass, including all six arrays, "
        "complete MT19937 state, next 16 random outputs, actual native stop "
        "metadata and exact [0,1,21] checkpoint schedule.",
        "",
        "The low-level observer keyword is present in the installed NumPy module "
        "but intentionally absent from the public fit_predict API; experiment "
        "helpers are not wheel contents.",
        "",
        "This is not a generic packaging smoke test replacement: the preexisting "
        "PCC regression workflow already tests installed Cython, Numba, NumPy "
        "on Windows/Linux Python 3.10/3.12.",
        "",
        "No guarantee of sandboxing, read-only callbacks, transactional exception "
        "semantics or overall RSS memory cap is implied.",
    ])
    args.output.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
