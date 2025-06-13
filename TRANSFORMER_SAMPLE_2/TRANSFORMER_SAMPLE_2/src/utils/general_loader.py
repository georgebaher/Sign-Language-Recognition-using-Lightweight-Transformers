import os
import numpy as np

def load_numpys_in_folder(folder, keyword2avoid):
    numpys_list = np.array([])
    for filename in sorted(os.listdir(folder)):
        if(keyword2avoid in filename):continue
        npfile = np.load(os.path.join(folder, filename))
        if (len(numpys_list) <= 0):
            numpys_list = npfile
        else:
            numpys_list = np.vstack((numpys_list, npfile))
    return numpys_list