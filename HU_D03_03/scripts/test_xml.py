"""Quick smoke test — verify XML compiles correctly and print joint info.

Usage (activate Vin env first):
    conda activate Vin
    python scripts/test_xml.py
"""

from pathlib import Path
import mujoco

XML = Path(__file__).parent.parent / "assets" / "robots" / "hu_d03" / "xmls" / "hu_d03.xml"


def main():
    print(f"Loading: {XML}")
    spec = mujoco.MjSpec.from_file(str(XML))
    m = spec.compile()

    print(f"\n✅ Compiled OK")
    print(f"   nq={m.nq}  nv={m.nv}  nu={m.nu}  nbody={m.nbody}  njnt={m.njnt}  neq={m.neq}")

    print("\n📐 Joints:")
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"   [{i:2d}] {name}")

    print("\n📍 Sites:")
    for i in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"   [{i:2d}] {name}")

    print("\n⛓️  Equality constraints:", m.neq)

    # Check foot sites exist
    left_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    if left_id >= 0 and right_id >= 0:
        print("\n✅ Foot sites: left_foot ✓  right_foot ✓")
    else:
        print(f"\n❌ Foot sites missing! left={left_id} right={right_id}")

    # Check IMU site
    imu_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    print(f"   IMU site: {'✓' if imu_id >= 0 else '❌ MISSING'}")


if __name__ == "__main__":
    main()
