from Feature_Processing.feature_selection.features.hand_pose_angles_features import TOP_ANGLE_BASES as hand_pose_angles
from Feature_Processing.feature_selection.features.hand_pose_landmarks_features import TOP_LANDMARKS_BASES as hand_pose_landmarks

# Define pose lower limb indices
lower_limb_pose_indices = set(str(i) for i in range(25, 33))

# Check if any P# landmarks fall in lower limb range
violating_landmarks = [f for f in hand_pose_landmarks if f.startswith("P#") and f.split("#")[1] in lower_limb_pose_indices]

if violating_landmarks:
    print("❌ Landmark features contain lower limb pose points:", violating_landmarks)
else:
    print("✅ Landmark features do NOT contain lower limb pose points.")

violating_angles = []

for angle in hand_pose_angles:
    if angle.startswith("P_Angle"):
        # Extract inside the {...}
        angle_body = angle.split("{")[1].rstrip("}")
        # Get each coordinate pair
        parts = angle_body.split("-")
        indices = []
        for part in parts:
            nums = part.replace("(", "").replace(")", "").split(",")
            for num in nums:
                indices.append(int(num.strip()))
        # Check if any number is in lower-limb pose indices
        if any(idx in lower_limb_pose_indices for idx in indices):
            violating_angles.append(angle)

if violating_angles:
    print("❌ Angle features contain lower limb pose points:", violating_angles)
else:
    print("✅ Angle features do NOT contain lower limb pose points.")