import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic and Drawing modules
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        # Convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect the landmarks
        results = holistic.process(image)

        # Draw the landmarks on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        if results.face_landmarks: # type: ignore
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks: # type: ignore
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks: # type: ignore
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )

        # Draw pose landmarks
        if results.pose_landmarks: # type: ignore
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # Display the resulting image
        cv2.imshow('MediaPipe Holistic', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
