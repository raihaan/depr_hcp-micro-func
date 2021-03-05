import os, sys
import numpy as np
import pandas as pd

#load left and right centroids
subj=sys.argv[1]
left_centroids = np.loadtxt(str(subj) + "/distance/" + str(subj) + ".glasser.left_centroid_vnum.txt")
right_centroids = np.loadtxt(str(subj) + "/distance/" + str(subj) + ".glasser.right_centroid_vnum.txt")

#right cortex is 0-180, left is 180-360, so only half of each of left/right above are needed
#make one array containing correct vertex num for whole brain
wb_centroids = np.zeros_like(left_centroids)
wb_centroids[0:180] = right_centroids[0:180]; wb_centroids[180:] = left_centroids[180:]

#load left vertex coords and right vertex coords (one line for every vertex)
left_vertex_coords = np.loadtxt(str(subj) + "/distance/" + str(subj) + ".left_coords.txt")
right_vertex_coords = np.loadtxt(str(subj) + "/distance/" + str(subj) + ".right_coords.txt")

#define centroid coord array as 360x3 - one line per parcel, 3 cols for xyz
centroid_coords = np.zeros((360,3)) 
#extract coords
for roi in range(0,180):
    centroid_coords[roi,:] = right_vertex_coords[int(wb_centroids[roi]),:]
for roi in range(180,360):
    centroid_coords[roi,:] = left_vertex_coords[int(wb_centroids[roi]),:]
    
#build distance matrix
euclid_mx = np.zeros((360,360)) #region by region

#cycle through each pair of regions, compute distance, store in euclid_mx
for r1 in range(0,360):
    for r2 in range(0,360):
        euclid_mx[r1,r2] = np.linalg.norm(centroid_coords[r1,:] - centroid_coords[r2,:])
        
#now save!
np.savetxt(str(subj) + "/distance/" + str(subj) + ".euclid_mx.txt",euclid_mx.astype('float32'),delimiter='\t',fmt='%f')
