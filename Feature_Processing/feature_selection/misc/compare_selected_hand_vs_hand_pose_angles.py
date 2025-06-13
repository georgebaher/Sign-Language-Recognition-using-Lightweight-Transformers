from Feature_Processing.feature_selection.features.hand_angles_features import TOP_ANGLE_BASES as HAND_BASES
from Feature_Processing.feature_selection.features.hand_pose_angles_features import TOP_ANGLE_BASES as HAND_POSE_BASES

hand_only_set = set(HAND_BASES)
combined_set = set(HAND_POSE_BASES)

# === Compare ===
missing = hand_only_set - combined_set
intersection = hand_only_set & combined_set
is_subset = hand_only_set.issubset(combined_set)

print("🧠 Hand-only angle features:", len(hand_only_set))
print("🧠 Combined hand+pose angle features (only hands):", len(combined_set))
print("🤝 Common hand features:", len(intersection))
print("❌ Missing hand-only features in combined:", len(missing))
print("✅ Is hand-only a subset of combined?", is_subset)

if missing:
    print("\n📊 Rank (index) of missing features from HAND_POSE_BASES:")
    for idx, feat in enumerate(HAND_BASES):
        if feat in missing:
            print(f"   {feat} → index {idx}")
