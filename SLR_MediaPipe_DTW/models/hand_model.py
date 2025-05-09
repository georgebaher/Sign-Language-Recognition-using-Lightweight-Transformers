from typing import List
import numpy as np
import mediapipe as mp


class HandModel(object):
    """
    Params
        landmarks: List of positions
    Args
        connections: List of tuples containing the ids of the two landmarks representing a connection
        feature_vector: List of length C * (C - 1) / 2 containing the angles between unique connection pairs
    """

    def __init__(self, landmarks: List[float]):
        self.connections = list(mp.solutions.holistic.HAND_CONNECTIONS)
        landmarks = np.array(landmarks).reshape((21, 3))
        self.feature_vector = self._get_feature_vector(landmarks)

    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        connections = self._get_connections_from_landmarks(landmarks)

        angles_list = []
        # print(connections)
        for i in range(len(connections)):
            for j in range(i + 1, len(connections)):
                u, v = connections[i], connections[j]
                # If either connection vector has -2 in it, skip (means missing)
                if np.any(u == -2) or np.any(v == -2):
                    angles_list.append(-2)
                else:
                    angle = self._get_angle_between_vectors(u, v)
                    angles_list.append(angle if angle == angle else -2)  # Handle NaN
        return angles_list

    def _get_connections_from_landmarks(self, landmarks: np.ndarray) -> List[np.ndarray]:
        vectors = []
        for a, b in self.connections:
            if np.any(landmarks[a] == -2) or np.any(landmarks[b] == -2):
                #print(f"Skipping connection ({a}, {b}) due to missing landmark")
                vectors.append(np.array([-2, -2, -2]))  # Placeholder for missing
            else:
                vectors.append(landmarks[b] - landmarks[a])
        return vectors

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm == 0:
            return -2  # Also pad with -2 for degenerate cases
        return np.arccos(np.clip(dot_product / norm, -1.0, 1.0))


if __name__ == "__main__":
    # Example: simulating a few missing landmarks with -2s
    test_landmarks = [i for i in range(63)]
    test_landmarks[6] = -2  # corrupt landmark 2 (x)
    test_landmarks[7] = -2  # corrupt landmark 2 (y)
    test_landmarks[8] = -2  # corrupt landmark 2 (z)

    print(test_landmarks)
    handModel = HandModel(landmarks=test_landmarks)
    print(f"# of connections: {len(handModel.connections)}")
    print(f"Feature vector length: {len(handModel.feature_vector)}")
    print(f"Missing angles: {handModel.feature_vector.count(-2)}")
