import json
import os
import random
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()
wlasl_metadata_path = os.getenv("WLASL_METADATA_PATH")
wlasl_videos_path = os.getenv("WLASL_VIDEOS_PATH")

def get_first_n_glosses_wlasl(n):
    """
    Loads the first `n` glosses and their existing video paths from the WLASL dataset.

    Returns:
        List[dict]: Each dict contains 'gloss' and list of valid 'videos_paths'.
    """
    with open(wlasl_metadata_path, 'r') as f:
        glosses = json.load(f)

    if n <= 0 or n > len(glosses):
        n = len(glosses)

    result = []
    for gloss in glosses[:n]:
        gloss_name = gloss["gloss"]
        video_ids = [instance['video_id'] for instance in gloss["instances"]]

        video_paths = [
            os.path.join(wlasl_videos_path, f"{video_id}.mp4")
            for video_id in video_ids
            if os.path.isfile(os.path.join(wlasl_videos_path, f"{video_id}.mp4"))
        ]

        if video_paths:
            result.append({
                "gloss": gloss_name,
                "videos_paths": video_paths
            })

    return result

def split_reference_test_glosses(glosses, test_ratio=0.3):
    """
    Splits each gloss's videos into training and test sets.

    Args:
        glosses (List[dict]): Output from get_first_n_glosses_wlasl
        test_ratio (float): Ratio of videos to use for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[List[str], List[str]]: train_paths, test_paths
    """
    test_paths = []
    references_paths = []
    # seed= random.randint(0, 10000)
    # random.seed(seed)

    for gloss_entry in glosses:
        videos = [(gloss_entry["gloss"], vid_path) for vid_path in gloss_entry["videos_paths"]]
        if len(videos) < 2:
            # Not enough samples to split
            continue

        references, test = train_test_split(videos, test_size=test_ratio, shuffle=True)
        test_paths.extend(test)
        references_paths.extend(references)

    return references_paths, test_paths


# Example usage
if __name__ == "__main__":
    glosses = get_first_n_glosses_wlasl(1) # eg: [{'gloss': 'book', 'videos_paths': ['C:/Users/georg/PycharmProjects/Acht/WLASL/start_kit/videos\\69241.mp4', ...}, {...}, ...]
    reference_videos, test_videos= split_reference_test_glosses(glosses)

    print("Test:", test_videos)
    print("References:", reference_videos)
