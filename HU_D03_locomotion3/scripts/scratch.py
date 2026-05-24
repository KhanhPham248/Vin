from mjlab.utils.lab_api.string import resolve_matching_names_values
import torch

joint_names = [f"joint_{i}" for i in range(55)] # Fake joint names
data_standing = {".*": 0.05}
data_walking = {".*": 0.05, "joint_1": 0.1}

a, b, c = resolve_matching_names_values(data_standing, joint_names)
print("standing len:", len(c))
try:
    a, b, c = resolve_matching_names_values(data_walking, joint_names)
    print("walking len:", len(c))
except Exception as e:
    print(e)
