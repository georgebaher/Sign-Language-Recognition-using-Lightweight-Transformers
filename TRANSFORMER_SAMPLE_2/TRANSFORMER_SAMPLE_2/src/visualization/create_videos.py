import cv2, os
import argparse
import src.utils.argutils as argutils
import math
import shutil


def create_video_from_imgs(imgs_folder, output_path, target_size=(320, 320), fps=25, filter_by_name =""):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    list_imgs = sorted(os.listdir(imgs_folder))
    if(not filter_by_name==""):
        filtered_imgs = filter(lambda k: filter_by_name in k, list_imgs)
    else:
        filtered_imgs = list_imgs
    for img_path in list(filtered_imgs):
        img = cv2.imread(os.path.join(imgs_folder,img_path))
        h,w,_ = img.shape
        if(w!=target_size[0] or h!=target_size[1]):
            resized_frame = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        else:
            resized_frame = img
        videoWriter.write(resized_frame)
    videoWriter.release()



def create_cuts_of_videos_ffmpeg(initial_time, end_time, in_video_path, out_video_path):
    # TODO: Check how to define better/more accurate timestamps to cut videos > right now really bad quality for short videos
    # Time format: 00:01:00 (hh:mm:ss)
    os.system("ffmpeg -i "+in_video_path+" -ss "+initial_time+" -to "+end_time+" -c copy "+out_video_path)
    #



def create_cuts_frames(initial_time, end_time, in_frames_path, out_frames_path, video_name, fps=25):
    os.makedirs(out_frames_path, exist_ok=True)
    ini_frame = math.floor(initial_time.total_seconds()*fps)+1 # we add 1 frame becasue we start in frame 1
    end_frame = math.ceil(end_time.total_seconds()*fps)+1
    # create list of frames
    list_frames_indx = [video_name+"_"+str(frameidx).zfill(6)+".jpg" for frameidx in list(range(ini_frame,end_frame))]
    for file2copy in list_frames_indx:
        ini_file = os.path.join(in_frames_path, file2copy)
        if(os.path.isfile(ini_file)):
            #shutil.copy2(ini_file, out_frames_path)
            cmd = "cp "+ini_file+" "+out_frames_path
            os.system(cmd)
        else:
            print("No frame: ", ini_file)
    return list_frames_indx, ini_frame, end_frame






def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # data
    parser.add_argument("--path_frames", type=str, required=True, help="Path with the frames of the videos")
    parser.add_argument("--filter_by_name", type=str, default='', help="keywords to select frames/images from the folder")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the generated video")
    parser.add_argument("--target_size", type=argutils.list_of_ints, default='320,320', help="Target size to create new frames")
    parser.add_argument("--fps", type=int, default=25, help="Fps of the video")
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_args()], add_help=False)
    args = parser.parse_args()

    ## CREATE VIDEOS FROM IMAGE FOLDER:
    create_video_from_imgs(args.path_frames, args.save_path, target_size=tuple(args.target_size), fps=args.fps, filter_by_name =args.filter_by_name)
    print("to do")

    # --path_frames
    # / run / user / 1000 / gvfs / smb - share: server = 137.250
    # .171
    # .20, share = datasets / AVASAG_dictionary / videos_glosses_from_sentences / 0001_0000
    # --save_path
    # / run / user / 1000 / gvfs / smb - share: server = 137.250
    # .171
    # .20, share = datasets / AVASAG_dictionary / tests / videos_check / 0001_0000.
    # mp4
    # --fps
    # 15
    # --target_size
    # 1920, 1080


