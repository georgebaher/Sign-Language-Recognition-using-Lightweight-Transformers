import os


if __name__ == '__main__':
    path_numbers = "/run/user/1000/gvfs/smb-share:server=137.250.171.20,share=datasets/AVASAG_dictionary/Fabrizzio_updated/mocapdata/num"

    list_files = os.listdir(path_numbers)

    extensions = ('.mp4')  # has to be tuple instead of list

    list_numbers = []
    for filename in sorted(list_files):
        if filename.endswith(extensions):
            #print(filename.split(".mp4")[0].zfill(3))
            list_numbers.append(filename.split(".mp4")[0].zfill(3))
    print(sorted(list_numbers))
