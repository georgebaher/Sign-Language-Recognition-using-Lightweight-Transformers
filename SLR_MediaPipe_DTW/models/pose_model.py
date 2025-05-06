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
        # Define the pose connections
        self.connections = list(mp.solutions.pose.POSE_CONNECTIONS)

        # Create feature vector (angles between unique connection pairs)
        landmarks = np.array(landmarks).reshape((33, 3))  # 33 pose landmarks
        self.feature_vector = self._get_feature_vector(landmarks)

    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        """
        Return list of angles between unique connection pairs (no self, no redundancy)
        """
        connections = self._get_connections_from_landmarks(landmarks)
        angles_list = []

        for i in range(len(connections)):
            for j in range(i + 1, len(connections)):
                angle = self._get_angle_between_vectors(connections[i], connections[j])
                angles_list.append(angle if angle == angle else 0)
        return angles_list

    def _get_connections_from_landmarks(self, landmarks: np.ndarray) -> List[np.ndarray]:
        """
        Create vector list from pose connections
        """
        return [landmarks[b] - landmarks[a] for a, b in self.connections]

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute angle between two 3D vectors
        """
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm == 0:
            return 0
        return np.arccos(np.clip(dot_product / norm, -1.0, 1.0))


if __name__ == "__main__":
    poseModel = PoseModel(landmarks=[f for f in range(100, 199)])  # 33 x 3 = 99 values
    print(f"# of pose connections: {len(poseModel.connections)}")
    print(f"Feature vector length: {len(poseModel.feature_vector)}")
    print(poseModel.feature_vector)
