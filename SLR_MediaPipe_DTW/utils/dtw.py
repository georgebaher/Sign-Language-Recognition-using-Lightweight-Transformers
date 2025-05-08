import pandas as pd
from fastdtw import fastdtw
from typing import List


def dtw_distances(recorded_embedding: List[List[float]], reference_signs: pd.DataFrame) -> pd.DataFrame:
    """
    Use DTW to compute similarity between a recorded embedding and reference embeddings.

    :param recorded_embedding: List of frames; each frame is a flat list of floats (e.g., angles or flattened landmarks)
    :param reference_signs: pd.DataFrame with columns:
                            - name: str
                            - embedding: List[List[float]]
                            - distance: float
    :return: DataFrame sorted by DTW distance (ascending)
    """

    for idx, row in reference_signs.iterrows():
        reference_embedding = row["embedding"]

        # DTW with Euclidean distance between frame vectors
        distance, _ = fastdtw(recorded_embedding, reference_embedding, dist=2)

        reference_signs.at[idx, "distance"] = distance

    return reference_signs.sort_values(by="distance").reset_index(drop=True)
