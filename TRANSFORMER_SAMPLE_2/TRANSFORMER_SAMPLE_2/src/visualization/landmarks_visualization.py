import os

import pandas as pd
import cv2
import argparse


###################### MEDIAPIPE CONNECTIONS & CONFIGURATION #################################
POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])
dict_PoseLandmark = {'NOSE': 0, 'LEFT_EYE_INNER':1,'LEFT_EYE':2,'LEFT_EYE_OUTER':3,'RIGHT_EYE_INNER':4,'RIGHT_EYE':5, 'RIGHT_EYE_OUTER':6,'LEFT_EAR':7,
                     'RIGHT_EAR':8, 'MOUTH_LEFT': 9, 'MOUTH_RIGHT':10, 'LEFT_SHOULDER':11, 'RIGHT_SHOULDER':12, 'LEFT_ELBOW':13, 'RIGHT_ELBOW':14, 'LEFT_WRIST':15, 'RIGHT_WRIST':16,
                     'LEFT_PINKY':17, 'RIGHT_PINKY':18, 'LEFT_INDEX':19, 'RIGHT_INDEX':20, 'LEFT_THUMB':21, 'RIGHT_THUMB':22, 'LEFT_HIP':23, 'RIGHT_HIP':24, 'LEFT_KNEE':25, 'RIGHT_KNEE':26,
                     'LEFT_ANKLE':27, 'RIGHT_ANKLE':28, 'LEFT_HEEL':29, 'RIGHT_HEEL':30, 'LEFT_FOOT_INDEX':31, 'RIGHT_FOOT_INDEX':32}

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])

handLandmarks = {"WRIST" : 0, "THUMB_CMC" :1, "THUMB_MCP" :2, "THUMB_IP" :3, "THUMB_TIP" :4, "INDEX_FINGER_MCP" :5, "INDEX_FINGER_PIP" :6, "INDEX_FINGER_DIP" :7, "INDEX_FINGER_TIP" :8,
  "MIDDLE_FINGER_MCP" :9, "MIDDLE_FINGER_PIP" :10, "MIDDLE_FINGER_DIP" :11,  "MIDDLE_FINGER_TIP" :12,  "RING_FINGER_MCP" :13,  "RING_FINGER_PIP" :14,  "RING_FINGER_DIP" :15, "RING_FINGER_TIP" :16,
  "PINKY_MCP" :17, "PINKY_PIP" :18,  "PINKY_DIP" :19,  "PINKY_TIP" :20}
##############################################################################################################




def plot_landmarks(image_path, landmarks_path, frame_idx=-1, save_path=""):
    if(frame_idx==-1):
        frame_idx = int(image_path.split("_", -1)[-1].split(".")[0])

    print("Processing ", str(frame_idx))
    # load the input image, resize it, and convert it to grayscale
    landmarks_df = pd.read_csv(landmarks_path, sep=",", header=0)
    landmark_df_i = landmarks_df.iloc[frame_idx-1]
    # Right Hand

    RHLandmarks = landmark_df_i.filter(regex=("RH[?!x,y]"))
    LHLandmarks = landmark_df_i.filter(regex=("LH[?!x,y]"))
    poseLandmarks = landmark_df_i.filter(regex=("P[?!x,y]"))

    # Check empty hands:
    totalEmptyhands = len(RHLandmarks) * -2
    if(sum(RHLandmarks)<=totalEmptyhands):
        print("   Not RH DETECTED")
    if (sum(LHLandmarks) <= totalEmptyhands):
        print("   Not LH DETECTED")

    image = cv2.imread(image_path)
    H, W, Ch = image.shape
    # loop over the landmarks
    # Right Hand (RED):
    Righthand_color = (0, 0, 255)
    RHandlandmarksValues = list(
        zip(RHLandmarks.filter(regex=("RH[?!x]")) * W, RHLandmarks.filter(regex=("RH[?!y]")) * H))
    image = plot_landmarks_connections(image, RHandlandmarksValues, HAND_CONNECTIONS, color=Righthand_color)

    for (x, y) in (RHandlandmarksValues):
        if (x < 0 or y < 0): continue
        cv2.circle(image, (int(x), int(y)), 3, Righthand_color, -1)

    # Left Hand (GREEN):
    Lefthand_color = (0, 0, 255)
    LHandlandmarksValues = list(
        zip(LHLandmarks.filter(regex=("LH[?!x]")) * W, LHLandmarks.filter(regex=("LH[?!y]")) * H))
    image = plot_landmarks_connections(image, LHandlandmarksValues, HAND_CONNECTIONS, color=Lefthand_color)

    for (x, y) in LHandlandmarksValues:
        if(x<0 or y<0):continue
        cv2.circle(image, (int(x), int(y)), 3, Lefthand_color, -1)

    # Body pose (BLUE):
    pose_color = (255, 0, 0)
    poselandmarksValues = list(
        zip(poseLandmarks.filter(regex=("P[?!x]")) * W, poseLandmarks.filter(regex=("P[?!y]")) * H))
    image = plot_landmarks_connections(image, poselandmarksValues, POSE_CONNECTIONS, color=pose_color)

    for (x, y) in poselandmarksValues:
        if(x==-2 or y==-2):continue
        cv2.circle(image, (int(x), int(y)),3, pose_color, -1)
    # show the output image with the face detections + facial landmarks
    if(save_path!=""):
        cv2.imwrite(save_path, image)
    else:
        cv2.imshow("Output", image)
        cv2.waitKey()




def plot_landmarks_connections(image, poselandmarksValues, connections, color, line_thickness = 2):
    H, W, Ch = image.shape
    for i,j in connections:
        if(sum(list(poselandmarksValues[i]))<0 or sum(list(poselandmarksValues[j]))<0):continue
        cv2.line(image, pt1=tuple(map(int, (poselandmarksValues[i]))), pt2=tuple(map(int, (poselandmarksValues[j]))), color=color, thickness=line_thickness)

    # cv2.imshow("Output", image)
    # cv2.waitKey()
    return image




def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # data
    parser.add_argument("--path_images", type=str, default="",
                        help="Path with the image to paint landmarks on")
    parser.add_argument("--path_landmarks", type=str, default="", help="Path with the landmarks of the video")
    parser.add_argument("--out_path_imgs", type=str, default="", help="Path to save images with keypoints")
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_args()], add_help=False)
    args = parser.parse_args()
    frame_idx = 1

    list_images = sorted(os.listdir(args.path_images))
    for image_name in list_images:
        plot_landmarks(os.path.join(args.path_images, image_name), args.path_landmarks, frame_idx, save_path=os.path.join(args.out_path_imgs,image_name))
        frame_idx+=1





