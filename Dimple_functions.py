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

def position_heatmap(df_data,df_dimples,bins=100, dimple_rad=40, dimple_dist=100):
    """plot a 2d historgram of particle positions around closest dimple
    Will only take particle data after running through find_nearest_dimple

    Args:
        df_data (pd dataframe): containging particle positions and nearest dimple colum
        df_dimples (pd dataframe): dimple data 
        bins (int): number of bins in 2d hist
        dimple_rad: dimple radius in pixels
        dimple_dist: dimple - dimple radial distance in pixels
    """
    x_dif=[]
    y_dif=[]
    for i in range(len(df_data)):
        dimple_id = df_data['Nearest_Dimple_ID'].iloc[i]
        x_dif.append(df_dimples['x'].iloc[dimple_id] - df_data['x'].iloc[i])
        y_dif.append(df_dimples['y'].iloc[dimple_id] - df_data['y'].iloc[i])

    plt.hist2d(x_dif,y_dif,bins=bins)
    circle1 = plt.Circle((0, 0), dimple_rad, color='r',fill=False)
    circle2 = plt.Circle((dimple_dist,0),dimple_rad, color='b', fill=False)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    angles = np.deg2rad(np.arange(0, 360, 60))  # 0,60,...,300 deg
    x_pts = dimple_dist * np.cos(angles)
    y_pts = dimple_dist * np.sin(angles)
    plt.scatter(x_pts, y_pts, c='b', s=30)
    plt.axvline(dimple_dist/2)
    plt.show()
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