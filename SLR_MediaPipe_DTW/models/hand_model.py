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
        # Define the connections
        self.connections = list(mp.solutions.holistic.HAND_CONNECTIONS)

        # Create feature vector (list of angles between unique connection pairs)
        landmarks = np.array(landmarks).reshape((21, 3))
        self.feature_vector = self._get_feature_vector(landmarks)

    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        """
        Params
            landmarks: numpy array of shape (21, 3)
        Return
            List of length C * (C - 1) / 2 containing all angles between unique connection pairs
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
        Params
            landmarks: numpy array of shape (21, 3)
        Return
            List of vectors representing hand connections
        """
        return [landmarks[b] - landmarks[a] for a, b in self.connections]

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        """
        Args
            u, v: 3D vectors representing two connections
        Return
            Angle between the two vectors
        """
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm == 0:
            return 0
        return np.arccos(np.clip(dot_product / norm, -1.0, 1.0))


if __name__ == "__main__":
    handModel = HandModel(landmarks=[f for f in range(100, 163)])
    print(f"# of connections: {len(handModel.connections)}")
    print(f"Feature vector length: {len(handModel.feature_vector)}")
    print(handModel.feature_vector)
