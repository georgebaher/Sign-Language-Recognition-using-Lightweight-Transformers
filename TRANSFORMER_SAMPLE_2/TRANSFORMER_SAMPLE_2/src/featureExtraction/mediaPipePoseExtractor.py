import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from google.protobuf.json_format import MessageToDict
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime
import argparse


def extract_XYZ(hands, pose, IMAGE_FILES, images_video_path, out_path_df, columns, padVal=-2):
    video_df = pd.DataFrame([], columns=columns + ["video", "frame", "path"])
    if(not os.path.exists(out_path_df)):
        for idx, file in enumerate(IMAGE_FILES):
            df_handsBody_coordinates = pd.DataFrame(padVal * np.ones(shape=(1, len(columns))), columns=columns)
            try:
                path_img = os.path.join(images_video_path, file)
                # For coding utf-8 filenames
                numpyarray = np.asarray(bytearray(open(path_img, "rb").read()), dtype=np.uint8)
                image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                # Convert the BGR image to RGB before utils.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                results_p = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Hands
                if results.multi_hand_landmarks is None:
                    print("no hands detected, -2 added to hands landmarks instead")

                else:
                    for iVal, hand_handedness in enumerate(results.multi_handedness):
                        handedness_dict = MessageToDict(hand_handedness)
                        handDetected = handedness_dict['classification'][0]["label"]
                        if (handDetected == "Left"):
                            hand2process = "LH"
                        else:
                            hand2process = "RH"
                            # print("Hand ", handDetected)

                        hand_landmarks = results.multi_hand_landmarks[iVal]  # Follow index of results.multi_handedness - NOT THE 'index' OF THE DICT
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y  # (1-y for providing same shape as in image)
                            z = hand_landmarks.landmark[i].z
                            # values+=[x, y, z]
                            df_handsBody_coordinates[hand2process + "x" + str(i)] = x
                            df_handsBody_coordinates[hand2process + "y" + str(i)] = y
                            df_handsBody_coordinates[hand2process + "z" + str(i)] = z

                # Pose
                if results_p.pose_landmarks is None:
                    print("no body detected, -2 added to body landmarks instead")
                else:
                    for i in range(len(results_p.pose_landmarks.landmark)):
                        x = results_p.pose_landmarks.landmark[i].x
                        y = results_p.pose_landmarks.landmark[i].y  # (1-y for providing same shape as in image)
                        z = results_p.pose_landmarks.landmark[i].z
                        df_handsBody_coordinates["Px" + str(i)] = x
                        df_handsBody_coordinates["Py" + str(i)] = y
                        df_handsBody_coordinates["Pz" + str(i)] = z

                video_df = video_df._append(pd.DataFrame([list(df_handsBody_coordinates.values[0]) + [
                     file.split("_")[0], (file.split(".")[0].split("_")[-1]), path_img]],
                                                         columns=columns + ["video", "frame", "path"]), ignore_index=True)

            except Exception:
                with open('logs.txt', 'a') as f:
                    print('extract_XYZ() ', images_video_path, ' ', idx + 1, ' ', file, file=f)
        # Save landmarks of the video
        video_df.to_csv(out_path_df, sep=",", header=True, index=False)



def main_extract_How2Sign(root_path, out_path_directory, hands, pose, columns):
    for video in os.listdir(root_path):  # os.listdir(root_path) #classes_df["video_id"].values
        #video = "{:05d}".format(int(video))
        images_video_path = os.path.join(root_path, video)
        if (os.path.isdir(images_video_path)):
            print("Processing ", video, " ...")
        if (os.path.isfile(os.path.join(out_path_directory, video + '_poses_landmarks.csv'))):
            # video was already processed
            continue
        IMAGE_FILES = sorted(os.listdir(images_video_path))

        out_path_df = os.path.join(out_path_directory, video + '_poses_landmarks.csv')
        extract_XYZ(hands, pose, IMAGE_FILES, images_video_path, out_path_df, columns)



def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # data
    parser.add_argument("--frames_path", type=str, default="",
                        help="Path containing the videos of the dataset to extract the frames from")
    parser.add_argument("--out_dir", type=str, default="", help="Path to save the generated frames")
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_args()], add_help=False)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                        min_detection_confidence=0.5)


    with open('logs.txt', 'a') as f:
        print('NEW RUN ', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), ' errors in this run:', file=f)

    columns1 = [["RHx" + str(i), "RHy" + str(i), "RHz" + str(i)] for i in range(21)]
    columns2 = [["LHx" + str(i), "LHy" + str(i), "LHz" + str(i)] for i in range(21)]
    columns3 = [["Px" + str(i), "Py" + str(i), "Pz" + str(i)] for i in range(33)]
    columns = np.concatenate((columns1, columns2), axis=0)
    columns = np.concatenate((columns, columns3), axis=0)
    columns = [item for sublist in columns for item in sublist]

    main_extract_How2Sign(args.frames_path, args.out_dir, hands, pose, columns)

