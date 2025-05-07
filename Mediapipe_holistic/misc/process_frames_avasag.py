import cv2
import mediapipe as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm

# Initialize MediaPipe Holistic and Drawing modules
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def get_first_n_instances_from_first_m_glosses_avasag(n, m):
    base_path = os.path.join("C:\\", "Users", "georg", "PycharmProjects", "Acht", "AVASAG",
                             "videos_glosses_from_sentences")
    all_folders = os.listdir(base_path)

    glosses=[]
    for i in range(1, 314):  # 0001 to 0313
        folder_first_part = f"{i:04d}"
        glosses.append({
            "gloss": folder_first_part,
            "instances" : [folder for folder in all_folders if folder.startswith(folder_first_part + "_")]
        })

    m=len(glosses) if m<0 or m>len(glosses) else m
    result = []
    for gloss in glosses[:m]:
        gloss_name = gloss["gloss"]
        instances = gloss["instances"]
        selected_instances = instances if n < 0 or n > len(instances) else instances[:n]
        result.append({
            "gloss": gloss_name,
            "folders": selected_instances,
        })
    return result

def process_images_in_folder(folder_path):
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"\033[93mWARNING: No image files found in {folder_path}.\033[0m")
        return

    # Create a window for display
    window_name = os.path.basename(folder_path)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 533, 400)
    cv2.moveWindow(window_name, 0, 0)

    with mp_holistic.Holistic(
        static_image_mode=True, # although static images
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5
    ) as holistic:
        for image_file in tqdm(image_files, desc="Processing Images", unit="image", colour="green"):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"\033[93mWARNING: Unable to read {image_path}. Skipping.\033[0m")
                continue

            # Convert the BGR image to RGB before processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # Draw landmarks on the image
            annotated_image = image.copy()

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Display the annotated image
            cv2.imshow(window_name, annotated_image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit early
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    glosses=get_first_n_instances_from_first_m_glosses_avasag(5, 5)
    for gloss in glosses:
        for instance in gloss["folders"]:
            images_folder_path = os.path.join("C:\\", "Users", "georg", "PycharmProjects", "Acht", "AVASAG", "videos_glosses_from_sentences",f"{instance}")
            process_images_in_folder(images_folder_path)
