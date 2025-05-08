from typing import List
import numpy as np
import mediapipe as mp


class PoseModel(object):
    """
    Params
        landmarks: List of 33 pose landmark positions (length 33*3 = 99)
    Args
        connections: List of tuples containing the ids of the two landmarks representing a connection
        feature_vector: List of length C * (C - 1) / 2 containing the angles between unique connection pairs
    """

    def __init__(self, landmarks: List[float]):
        self.connections = list(mp.solutions.pose.POSE_CONNECTIONS)

        landmarks = np.array(landmarks).reshape((33, 3))  # 33 pose landmarks
        self.feature_vector = self._get_feature_vector(landmarks)

    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        connections = self._get_connections_from_landmarks(landmarks)

        angles_list = []
        for i in range(len(connections)):
            for j in range(i + 1, len(connections)):
                u, v = connections[i], connections[j]
                if np.any(u == -2) or np.any(v == -2):
                    angles_list.append(-2)
                else:
                    angle = self._get_angle_between_vectors(u, v)
                    angles_list.append(angle if angle == angle else -2)
        return angles_list

    def _get_connections_from_landmarks(self, landmarks: np.ndarray) -> List[np.ndarray]:
        vectors = []
        for a, b in self.connections:
            if np.any(landmarks[a] == -2) or np.any(landmarks[b] == -2):
                #print(f"Skipping connection ({a}, {b}) due to missing landmark")
                vectors.append(np.array([-2, -2, -2]))
            else:
                vectors.append(landmarks[b] - landmarks[a])
        return vectors

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm == 0:
            return -2
        return np.arccos(np.clip(dot_product / norm, -1.0, 1.0))


if __name__ == "__main__":
    # Create valid pose input and corrupt landmark 5
    pose_landmarks = [f for f in range(100, 199)]  # 33 x 3 = 99
    pose_landmarks[15] = -2  # x of landmark 5
    pose_landmarks[16] = -2  # y of landmark 5
    pose_landmarks[17] = -2  # z of landmark 5

    poseModel = PoseModel(landmarks=pose_landmarks)
    print(f"# of pose connections: {len(poseModel.connections)}")
    print(f"Feature vector length: {len(poseModel.feature_vector)}")
    print(f"Missing angles: {poseModel.feature_vector.count(-2)}")
