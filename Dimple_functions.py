import numpy as np
import pandas as pd
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

def position_heatmap(df_data,df_dimples,bins=100, dimple_rad=40, dimple_dist=100, pixpermm= 34.5):
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

def plot_in_dimple_ratio(plate, fill):

    #read plate dict
    plateD = "Y:\\Joe_shaker1\\dimple_exp\\plate_dict.txt"
    with open(plateD, 'r') as file:
        exec(file.read())


    load_path = "C://Users//pczjo//OneDrive - The University of Nottingham//Desktop//dimple_exp//DFs//plate"+ str(plate) + "//" + str(fill) + "g.pkl"
    
    dimple_rad = plates[plate]["diameter"]/2 #in pixels
    dimple_dist = 4.97  ##measure in mm (same for all paltes)
    
    pix_per_mm = plates[plate]["scale"] #pixel to mm conversion factor, calculated from image of scalebar
    
    if plate == 3 and fill == 91:
        dimple_rad = plates[391]["diameter"]/2 
        pix_per_mm = plates[391]["scale"]
    
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    DCs = list(data.keys())
    
    ratio = np.zeros(len(DCs))  # Initialize an array to store the ratios
    
    for i in range(len(DCs)):
        DC = str(DCs[i])
        data_mm = data[DC]['Distance_to_Nearest'] / pix_per_mm
        ratio[i] = (data_mm <= dimple_rad).mean()  # Proportion of particles within the dimple radius

    i = np.where(ratio < 0.9)[0]
    t_dc = DCs[i[0]]
    print("Transition DC: ", t_dc)

    plt.plot(DCs, ratio, label=("plate " + str(plate) + " fill " + str(fill) + "g"))
    plt.xlabel('DC')
    plt.ylabel('Proportion of Particles in Dimple')
    plt.xticks(DCs[1::2])
    plt.legend()

def plot_crystal_factor(folder, acc = False):
    """Plot crystal factor for a range of tracked videos,
        files must be named xxx.hdf5 where xxx is the DC of the video, and be in the same folder.
        videos must have been tracked and assinged a boolean 'crystal' column using hexatic postprocess. 

    Args:
        folder: full location e.g. "Y:\\Joe_shaker1\\dimple_exp\\hexatic\\plate2\\75g\\"
        acc (bool, optional): if True, uses meta data to plot gamma instead of DC. Defaults to False.
    """
    #plt.figure()
    files = glob.glob(folder+"[0-9][0-9][0-9].hdf5")
    DC = np.zeros(len(files))
    crystal_factor = np.zeros(len(files))
    for i in range(len(files)):
        file = files[i]
        data = pd.read_hdf(file)
        DC[i] = int(file[-8:-5])
        crystal_factor[i] = data['crystal'].mean()
        #print(crystal_factor[i], file[-8:])

    if acc:
        print('acc connected')
        meta = np.loadtxt(folder+"MetaData.txt", unpack=True, skiprows=2, usecols=(2,4))
        meta_dc, acc = meta
        meta_dc_int = [int(x) for x in meta_dc]
        #print(meta_dc_int)
        average_acc = np.zeros(len(DC))
        print(DC)

        for i, dc in enumerate(DC):
            mask = (meta_dc_int == dc)
            matching_accs = meta[1][mask]
            average_acc[i] = np.mean(matching_accs)
            #print(average_acc[i], dc)
        print(average_acc)
        plt.scatter(average_acc, crystal_factor)
        plt.xlabel('g')
        plt.ylabel('Crystal Factor')
        plt.title('Crystal Factor vs g for ' + folder[-12:-1])

    else:
        plt.scatter(DC, crystal_factor)
        plt.xlabel('DC')
        plt.ylabel('Crystal Factor')
        plt.title('Crystal Factor vs DC for ' + folder[-12:-1])
        plt.xticks(DC[0::10])
    
    plt.grid()
    return()

def plot_crystal_ramp(file):#, start, stop, acc = False):
    """Plot crystal factor for each frame in a video hdf5
        Video must be tracked and assigned crystal column
    Args:
        file: full location e.g. "Y:\\Joe_shaker1\\dimple_exp\\hexatic\\plate2\\75g\\600.hdf5"
        acc (bool, optional): if True, uses meta data to plot gamma instead of DC. Defaults to False.
    """
    data = pd.read_hdf(file)
    number_frames = data.index.nunique()
    print(number_frames)
    crystal_factors = []
    times = []
    frame_rate = 1/20
    for frame in range(number_frames):
        crystal_factor = data.loc[frame]['crystal'].mean()
        crystal_factors.append(crystal_factor)
        time = frame * frame_rate
        times.append(time)
    plt.plot(times,crystal_factors)
    plt.xlabel('Time (s)')
    plt.ylabel('Crystal Factor')
    plt.title('Crystal Factor vs Time for ' + file[-12:-5])
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