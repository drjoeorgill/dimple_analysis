import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from labvision.video import ReadVideo
from labvision.images import display, gaussian_blur, save
from labvision.images.colours import bgr_to_gray

import cv2

def average_vid(file_name, display_im = 'No', save_im = 'No'):
    """convert file_name.mp4 into an averaged image

    Args:
        file_name (str): mp4 file of video to be averaged (no .mp4 required)
        display_im (str, optional): if Yes, average image is displayed. Defaults to 'No'.
        save_im (str, optional): if Yes, average image is saved under the same filename as video (.png). Defaults to 'No'.
    """
    vid_name = file_name + ".mp4"
    pic_name = file_name + ".png"
    vid=ReadVideo(vid_name)
    for i , image in enumerate(vid):
        gimage = bgr_to_gray(image)
        if i == 0:
            total_image = np.float64(gaussian_blur(gimage.copy(),kernel=(3,3)))
        else:
            total_image = total_image + np.float64(gaussian_blur(gimage,kernel=(3,3)))
    
    print(vid.num_frames)
    if display_im == 'Yes':
        display(total_image/(vid.num_frames*256))
    if save_im == 'Yes':
        save(total_image/vid.num_frames, pic_name)
    return()

def find_nearest_dimple(df_data,df_dimples):
    df_data_new = df_data
    particle_coords = df_data_new[['x', 'y']].values

    tree = KDTree(df_dimples[['x', 'y']].values)

    distances, indices = tree.query(particle_coords)

    nearest_point_ids = df_dimples.iloc[indices]['particle'].values

    df_data_new['Nearest_Dimple_ID'] = nearest_point_ids
    df_data_new['Distance_to_Nearest'] = distances
    return()


if __name__ == "__main__":
    file_name = "625_Trim"
    vid_name = file_name + ".mp4"
    pic_name = file_name + "average.png"
    vid=ReadVideo(vid_name)
    for i , image in enumerate(vid):
        gimage = bgr_to_gray(image)
        if i == 0:
            #print(image)
            #sz = np.shape(image[:,:,0])
            total_image = np.float64(gaussian_blur(gimage.copy(),kernel=(3,3)))
        else:
            total_image = total_image + np.float64(gaussian_blur(gimage,kernel=(3,3)))
    
    print(vid.num_frames)

    display(total_image/(vid.num_frames*256))

    save(total_image/vid.num_frames, pic_name)