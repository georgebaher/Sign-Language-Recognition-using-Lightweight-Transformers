# calculate_angles.py

import pandas as pd
import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm
from connections import get_connections


class LandmarkModel:
    def __init__(self, landmarks, connections, num_landmarks):
        self.connections_map = connections
        self.num_landmarks = num_landmarks
        landmarks_array = np.array(landmarks, dtype=float).reshape((self.num_landmarks, 2))
        self.feature_vector = self._get_feature_vector(landmarks_array)

    def _get_connections_from_landmarks(self, landmarks):
        vectors = []
        for a, b in self.connections_map:
            if np.any(landmarks[a] == -2) or np.any(landmarks[b] == -2):
                vectors.append(np.array([-2, -2]))
            else:
                vectors.append(landmarks[b] - landmarks[a])
        return vectors

    @staticmethod
    def _get_angle_between_vectors(u, v):
        if np.all(u == -2) or np.all(v == -2): return -2
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm == 0: return -2
        angle_rad = np.arccos(np.clip(dot_product / norm, -1.0, 1.0))
        return angle_rad

    def _get_feature_vector(self, landmarks):
        connection_vectors = self._get_connections_from_landmarks(landmarks)
        num_connections = len(connection_vectors)
        angles = [self._get_angle_between_vectors(connection_vectors[i], connection_vectors[j])
                  for i in range(num_connections) for j in range(i + 1, num_connections)]
        return angles


def get_header(connections):
    conns_as_tuples = [tuple(sorted(c)) for c in connections]
    angle_labels = [f"Angle{{{conns_as_tuples[i]}-{conns_as_tuples[j]}}}"
                    for i in range(len(conns_as_tuples)) for j in range(i + 1, len(conns_as_tuples))]
    return angle_labels


def main():
    parser = ArgumentParser(description="Calculate angular features from landmark data using column indices.")
    parser.add_argument('-i', '--input_path', type=str, required=True, help="Path to the input landmark Parquet file.")
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help="Path to save the output angles Parquet file.")
    parser.add_argument('--source', type=str, required=True, choices=['mediapipe', 'vitpose'],
                        help="The source skeleton ('mediapipe' or 'vitpose').")
    parser.add_argument('--body_part', type=str, required=True, choices=['pose', 'hand'],
                        help="The body part to process ('pose' or 'hand').")
    parser.add_argument('--num_landmarks', type=int, required=True,
                        help="Number of landmarks PER SET (e.g., 21 for one hand, 25 for upper-body pose).")
    parser.add_argument('--start_col', type=int, required=True, help="The starting column index of the landmark data.")
    args = parser.parse_args()

    print(f"--- Starting Angle Calculation for '{args.source} {args.body_part}'  (using column indices) ---")

    df = pd.read_parquet(args.input_path)

    full_connections = get_connections(args.source, args.body_part)
    connections = [
        conn for conn in full_connections
        if conn[0] < args.num_landmarks and conn[1] < args.num_landmarks
    ]
    print(f"Using {len(connections)} connections valid for {args.num_landmarks} landmarks.")

    all_frames_angles = []
    angle_headers = get_header(connections)
    is_hand = args.body_part == 'hand'

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Frames", unit="frame"):
        frame_angles = {'video_id': row['video_id'], 'gloss': row['gloss']}

        if is_hand:
            num_coords_per_hand = args.num_landmarks * 2
            left_landmarks = row.iloc[args.start_col: args.start_col + num_coords_per_hand].values
            right_landmarks = row.iloc[
                              args.start_col + num_coords_per_hand: args.start_col + (2 * num_coords_per_hand)].values

            l_model = LandmarkModel(left_landmarks, connections, args.num_landmarks)
            r_model = LandmarkModel(right_landmarks, connections, args.num_landmarks)

            l_prefixed_header = [f"L_{h}" for h in angle_headers]
            r_prefixed_header = [f"R_{h}" for h in angle_headers]

            frame_angles.update(zip(l_prefixed_header, l_model.feature_vector))
            frame_angles.update(zip(r_prefixed_header, r_model.feature_vector))
        else:
            num_coords = args.num_landmarks * 2
            landmarks = row.iloc[args.start_col: args.start_col + num_coords].values

            model = LandmarkModel(landmarks, connections, args.num_landmarks)
            frame_angles.update(zip(angle_headers, model.feature_vector))

        all_frames_angles.append(frame_angles)

    angles_df = pd.DataFrame(all_frames_angles).fillna(-2)
    angles_df.to_parquet(args.output_path, index=False)
    print(f"Frame-by-frame angles saved to: {args.output_path}")


if __name__ == "__main__":
    main()