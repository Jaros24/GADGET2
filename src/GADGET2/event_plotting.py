"""
Functions Related to Visualizing Individual Events
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


def show_plot(h5_dir, event_num, xHit, yHit, zHit, eHit):
    event_num = int(event_num)
    index = self.run_data.get_index(event_num)
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-35, 35)
    ax.set_ylim3d(-35, 35)
    ax.set_zlim3d(0, 35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"3D Point-cloud of Event {event_num}", fontdict = {'fontsize' : 10})
    ax.scatter(xHit, yHit, zHit-np.min(zHit), c=eHit, cmap='RdBu_r')
    cbar = fig.colorbar(ax.get_children()[0])
    plt.show(block=False)

def track_w_trace(self):
    event_num = int(self.event_num_entry.get())
    index = self.run_data.get_index(event_num)
    plt.figure()
    self.run_data.make_image(index, show=True)
    
def track_w_trace_raw(self):
    event_num = int(self.event_num_entry.get())
    index = self.run_data.get_index(event_num)
    plt.figure()
    self.run_data.make_image(index, use_raw_data = True, show=True)
    
def track_w_trace_raw_smooth(self):
    event_num = int(self.event_num_entry.get())
    index = self.run_data.get_index(event_num)
    plt.figure()
    self.run_data.make_image(index, use_raw_data = True, show=True, smoothen = True)

def show_point_cloud(self):
    event_num = int(self.event_num_entry.get())
    xHit, yHit, zHit, eHit = self.run_data.get_hit_lists(event_num)
    self.show_plot(xHit, yHit, zHit, eHit)

def plot_dense_3d_track(self):
    radius = 5

    event_num = int(self.event_num_entry.get())
    xHit, yHit, zHit, eHit = self.run_data.get_hit_lists(event_num)
    
    nbrs = NearestNeighbors(radius=radius).fit(np.vstack((xHit, yHit, zHit)).T)

    # Find the points within the specified radius
    points_within_radius = nbrs.radius_neighbors(np.vstack((xHit, yHit, zHit)).T, return_distance=False)

    # Interpolate between points within the radius
    xHit_dense, yHit_dense, zHit_dense, eHit_dense = [], [], [], []
    for i, neighbors in enumerate(points_within_radius):
        for j in neighbors:
            t = np.random.rand()
            xHit_dense.append(xHit[i] + t * (xHit[j] - xHit[i]))
            yHit_dense.append(yHit[i] + t * (yHit[j] - yHit[i]))
            zHit_dense.append(zHit[i] + t * (zHit[j] - zHit[i]))
            eHit_dense.append(eHit[i] + t * (eHit[j] - eHit[i]))

    # Convert lists to arrays
    xHit_dense = np.array(xHit_dense)
    yHit_dense = np.array(yHit_dense)
    zHit_dense = np.array(zHit_dense)
    eHit_dense = np.array(eHit_dense)

    self.show_plot(xHit_dense, yHit_dense, zHit_dense, eHit_dense)