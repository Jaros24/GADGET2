from matplotlib.path import Path
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .run_h5 import GadgetRunH5
import datetime
import random
import os
import pickle
from tqdm import tqdm
import io
from PIL import Image


def get_RvE_cut_indexes(run_data:GadgetRunH5, points):
    """
    points: list of (energy, range) tuples defining a cut in RvE
    Energy is in MeV, range in mm
    """
    path = Path(points)
    to_return = []
    index = 0
    while index < len(run_data.good_events):
        this_point = (run_data.total_energy_MeV[index], run_data.len_list[index])
        if path.contains_point(this_point):
            to_return.append(index)
        index += 1
    return to_return

def plot_RVE(run_data:GadgetRunH5, bins:list=[200, 200], fig_name='RvE', clear=True, show=True, log_scale=True):
    num_range_bins, num_energy_bins = bins
    
    plt.figure(fig_name, clear=clear)
    plt.xlabel('Energy (MeV)', fontdict={'fontsize': 20})
    plt.ylabel('Range (mm)', fontdict={'fontsize': 20})
    plt.title(f'Range vs Energy \n Energy Bins = {num_energy_bins} | Range Bins = {num_range_bins}', fontdict={'fontsize': 20})
    tot_energy_temp = np.concatenate(([0], run_data.total_energy_MeV))
    len_list_temp = np.concatenate(([0], run_data.len_list))
    if log_scale:
        norm = colors.LogNorm()
    else:
        norm = colors.Normalize()
    plt.hist2d(tot_energy_temp, len_list_temp, (num_energy_bins, num_range_bins), 
               cmap=plt.cm.jet, norm=norm)
    plt.colorbar()
    plt.gca().set_facecolor('darkblue')
    if show:
        plt.show(block=False)

def show_RVE_event(run_data:GadgetRunH5, event_num:int): #TODO: add "show annotation" checkbox
    #only draw plot if it's not already open
    if not plt.fignum_exists('RvE'):
        plot_RVE(run_data, show=False)
    else:
        plt.figure('RvE')  # switch focus back to RvE plot)
    event_index = run_data.get_index(event_num)
    plt.plot(run_data.total_energy_MeV[event_index], run_data.len_list[event_index], 'ro', picker=5) 
    plt.annotate(f"Evt#: {event_num}", (run_data.total_energy_MeV[event_index], 
                run_data.len_list[event_index]), textcoords="offset points", xytext=(-15,7),
                ha='center', fontsize=10, color='black',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", edgecolor="black"))
    plt.show(block=False)

def save_cut_files(run_data:GadgetRunH5, points, use_raw_data=False):
    now = datetime.datetime.now()
    rand_num = str(random.randrange(0,1000000,1))
    cut_name = rand_num+now.strftime("CUT_Date_%m_%d_%Y")
    if use_raw_data:
        cut_name += '_raw'
    imageCut_path = os.path.join(run_data.folder_path, cut_name)
    print('NEW DIRECTORY', imageCut_path)

    # save an image for future cut selection
    plot_RVE(fig_name=cut_name)
    ax = plt.gca()
    #add once point and use this to close the path
    #actual value of final vertex is ignored for CLOSEPOLY code
    points_list = list(points)
    points_list.append([0,0])
    codes = [Path.LINETO]*len(points_list)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    path = Path(points_list, codes)
    
    to_draw = patches.PathPatch(path, fill=False, color='red')
    ax.add_patch(to_draw)
    plt.savefig(os.path.join(run_data.folder_path, cut_name+'.jpg'))
    plt.close()
	
    os.makedirs(imageCut_path)
	
	# Process images in chunks to avoiding overloading memory
	# Bhavya
    cut_indices = run_data.get_RvE_cut_indexes(points)
    chunk_size = 500
    num_images = len(cut_indices)
    print("Total Number of Image:", num_images)
    num_chunks = (num_images + chunk_size - 1) // chunk_size
    print("Total Number of Chunks:", num_chunks)
    chunk_num = 1

    pbar = tqdm(total=num_chunks)
    for chunk_idx in range(num_chunks):
        print(f"Processing Chunk {chunk_idx+1} of {num_chunks}")
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_images)

        chunk_indices = cut_indices[start_idx:end_idx]
        image_data = run_data.save_cutImages(chunk_indices, use_raw_data=use_raw_data)

        my_dpi = 96
        fig_size = (224/my_dpi, 73/my_dpi)  # Fig size to be used in the main thread
        fig, ax = plt.subplots(figsize=fig_size)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

		# Plot and save images in the main thread
        for pad_plane, trace, title, filename in image_data:
			# Plot trace
			# Plot trace
            ax.clear()
            x = np.linspace(0, len(trace)-1, len(trace))
            ax.fill_between(x, trace, color='b', alpha=1)

            # Assuming pad_plane is a NumPy array with shape (height, width, 3)
            alpha_channel = np.ones((pad_plane.shape[0], pad_plane.shape[1], 1), dtype=pad_plane.dtype) * 255
            pad_plane_rgba = np.concatenate((pad_plane, alpha_channel), axis=2)
            # Now pad_plane_rgba has an additional alpha channel and can be concatenated with trace_img

            # Ensure trace image is saved as PNG and read it
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=my_dpi)
            buf.seek(0)
            with Image.open(buf) as im:
                trace_img_png = np.array(im)
            buf.close()

            # Concatenate pad_plane_rgba and trace_img_png
            complete_image = np.append(pad_plane_rgba, trace_img_png, axis=0)

            # Convert image data to uint8 if it's not already
            if complete_image.dtype != np.uint8:
                if complete_image.max() > 1:
                    complete_image = complete_image.astype(np.uint8)
                else:
                    complete_image = (255 * complete_image).astype(np.uint8)

            # Save the final concatenated image as PNG
            plt.imsave(os.path.join(imageCut_path, filename), complete_image)
            # Close the figure to free memory
            plt.close(fig)


            chunk_num += 1
            pbar.update(n=1)

		# Update the GUI and process pending events
        # root.update_idletasks()
        # root.update()

    print("All images have been processed")
	
	# Pickle cut_indices
    cut_indices_H5list = run_data.good_events[cut_indices]
    cut_indices_str = f"cut_indices_H5list.pkl"
    cut_indices_path = os.path.join(imageCut_path, cut_indices_str)
    with open(cut_indices_path, "wb") as file:
            pickle.dump(cut_indices_H5list, file)
