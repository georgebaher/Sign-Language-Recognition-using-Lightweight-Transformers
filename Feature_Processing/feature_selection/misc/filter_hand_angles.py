from Feature_Processing.feature_selection.features.hand_angles_features import TOP_ANGLE_BASES as HAND_BASES
from Feature_Processing.feature_selection.features.hand_pose_angles_features import TOP_ANGLE_BASES as HAND_POSE_BASES

# Full sets
hand_only_set = set(HAND_BASES)
combined_set = set(HAND_POSE_BASES)

# Find missing and filtered
missing = hand_only_set - combined_set
filtered_hand_angles = [f for f in HAND_BASES if f not in missing]

print("🔢 Filtered hand angles (present in both):", len(filtered_hand_angles))
print("❌ Removed (missing) hand angles:", len(missing))

# === Save to Python file ===
output_path = "filtered_hand_angles_features.py"
with open(output_path, "w") as f:
    f.write("# Auto-generated filtered hand angle features\n")
    f.write("TOP_ANGLE_BASES = [\n")
    for feat in filtered_hand_angles:
        f.write(f'    "{feat}",\n')
    f.write("]\n")

print(f"\n💾 Saved to {output_path}")
