#!/usr/bin/env python3

import argparse
import os
import sys
import time


def _default_xml_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "xml", "HU_D03_test_minimal.xml")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a MuJoCo MJCF (.xml) model and display it in the MuJoCo viewer."
    )
    parser.add_argument(
        "--xml",
        default=_default_xml_path(),
        help="Path to MJCF XML file (default: HU_D03_description/xml/HU_D03_test_minimal.xml)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep to approximately match the model timestep.",
    )
    args = parser.parse_args()

    xml_path = os.path.abspath(args.xml)
    if not os.path.exists(xml_path):
        print(f"ERROR: XML not found: {xml_path}", file=sys.stderr)
        return 2

    try:
        import mujoco
        from mujoco import viewer
    except Exception as exc:  # ImportError or GL import errors
        print("ERROR: Failed to import MuJoCo Python package ('mujoco').", file=sys.stderr)
        print("Install (typical): pip install mujoco", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        return 3

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as exc:
        print(f"ERROR: Failed to load MJCF: {xml_path}", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        return 4

    data = mujoco.MjData(model)

    print(f"Loaded: {xml_path}")
    print("Close the viewer window to exit.")

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            if args.realtime:
                time.sleep(model.opt.timestep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
