#!/bin/bash

# ====================================================================== #
# 2021_03_12 Raihaan Patel
# Creates individual command list file that is submitted by qbatch
# run per subject: for subj in ../raw_data/structural/??????; do ./preproc.sh $subj; done
# can then do the following to submit joblists for each subj: 
# for file in ??????/*joblist; do echo bash $file; done > preproc_joblist
# and
# module load cobralab/2019b
# qbatch -w 00:30:00 -c 50 preproc_joblist
#
# Performs commands needed to parcellate rsfc surface data according to an atlas
# Mean rsfc in each region is taken
# vertices weighted according to vertex area - lareger area gets more weight
# outputs pscalar.nii file containing mean rsfc in each region, and .txt file containing the vals per region for more general processing
#
# Performs commands needed to parcellate t1t2 surface data according to an atlas
# Mean t1t2 in each region is taken
# vertices weighted according to vertex area - lareger area gets more weight
# outputs pscalar.nii file containing mean t1t2 in each region, and .txt file containing the vals per region for more general processing
#
# Performs commands needed to identify vertex coordinates of each region centroid
# computes euclid distance between each centrtoid
# outputs regionxregion matrix in .txt format containing distances
#
# ====================================================================== #

#set vars, just 1 var - input id
input=$1
subj=$(basename $input)
rsfc_out_dir="${subj}/rsfc/"
joblist="${subj}/${subj}.joblist"
#make output dir 
mkdir -p $rsfc_out_dir

#set some paths to raw data
raw_func_dir="../raw_data/functional/${subj}/MNINonLinear/Results/"
raw_struct_dir="../raw_data/structural/"
#set path to dir with cifti atlases for parcellating
atlas_dir="../raw_data/atlas_files/"



#define input files
atlas_file="${atlas_dir}/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
left_surface="${raw_struct_dir}/${subj}/T1w/fsaverage_LR32k/${subj}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
right_surface="${raw_struct_dir}/${subj}/T1w/fsaverage_LR32k/${subj}.R.midthickness_MSMAll.32k_fs_LR.surf.gii"

touch ${joblist}

#compute mean time series of each parcel
#use weighted avg based on surface file/vertex area
#create rsfc mx from parcellated tseries (parcel-parcel connectivity mx)
#convert to txt for general analysis
for rs_run in {rfMRI_REST1_LR,rfMRI_REST1_RL}
do
echo wb_command -cifti-parcellate ${raw_func_dir}/${rs_run}/${rs_run}_Atlas_MSMAll_hp2000_clean.dtseries.nii ${atlas_file} COLUMN ${rsfc_out_dir}/${subj}.${rs_run}.ptseries.nii -spatial-weights -left-area-surf ${left_surface} -right-area-surf ${right_surface} >> ${joblist}
echo wb_command -cifti-correlation -mem-limit 2 ${rsfc_out_dir}/${subj}.${rs_run}.ptseries.nii ${rsfc_out_dir}/${subj}.${rs_run}.netmat.pconn.nii -fisher-z >> ${joblist}
echo wb_command -cifti-convert -to-text ${rsfc_out_dir}/${subj}.${rs_run}.netmat.pconn.nii ${rsfc_out_dir}/${subj}.${rs_run}.netmat.txt >> ${joblist}
done

echo "" >> ${joblist}
#average LR and RL time series to compute 1 netmat per run
echo wb_command -cifti-average ${rsfc_out_dir}/${subj}.rfMRI_REST1_avg.netmat.pconn.nii -cifti ${rsfc_out_dir}/${subj}.rfMRI_REST1_LR.netmat.pconn.nii -cifti ${rsfc_out_dir}/${subj}.rfMRI_REST1_RL.netmat.pconn.nii >> ${joblist}
echo wb_command -cifti-convert -to-text ${rsfc_out_dir}/${subj}.rfMRI_REST1_avg.netmat.pconn.nii ${rsfc_out_dir}/${subj}.rfMRI_REST1.netmat.txt >> ${joblist}
echo "" >> ${joblist}

echo wb_command -cifti-average ${rsfc_out_dir}/${subj}.rfMRI_REST2_avg.netmat.pconn.nii -cifti ${rsfc_out_dir}/${subj}.rfMRI_REST2_LR.netmat.pconn.nii -cifti ${rsfc_out_dir}/${subj}.rfMRI_REST2_RL.netmat.pconn.nii >> ${joblist}
echo wb_command -cifti-convert -to-text ${rsfc_out_dir}/${subj}.rfMRI_REST2_avg.netmat.pconn.nii ${rsfc_out_dir}/${subj}.rfMRI_REST2.netmat.txt >> ${joblist}

echo "" >> ${joblist}
echo "#now running t1t2 preproc" >> ${joblist}
echo "" >> ${joblist}

#now t1t2
t1t2_out_dir="${subj}/t1t2/"
#make output dir 
mkdir -p $t1t2_out_dir

#define input files
raw_t1t2="${raw_struct_dir}/${subj}/MNINonLinear/fsaverage_LR32k/${subj}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii"
parcellated_t1t2="${t1t2_out_dir}/${subj}.weighted.parcellated.t1t2.pscalar.nii"
parcellated_t1t2_txt="${t1t2_out_dir}/${subj}.weighted.parcellated.t1t2.txt"

#parcellate the raw t1t2 file according to Glasser atlas
#take the mean of t1t2 vals in each parcels
#weight vertices according to their vertex area -> larger vertices contribute more
echo wb_command -cifti-parcellate ${raw_t1t2} ${atlas_file} COLUMN ${parcellated_t1t2} -method MEAN -spatial-weights -left-area-surf ${left_surface} -right-area-surf ${right_surface} >> ${joblist}

#add a space betwn commands because im blind
echo "" >> ${joblist} 

#convert to .txt file for python processing
echo wb_command -cifti-convert -to-text ${parcellated_t1t2} ${parcellated_t1t2_txt} >> ${joblist}

echo "" >> ${joblist}
echo "#now running distance preproc" >> ${joblist}
echo "" >> ${joblist}

distance_out_dir="${subj}/distance/"
#make output dir 
mkdir -p $distance_out_dir

#get coordinates of left and right surface vertices
echo wb_command -surface-coordinates-to-metric ${left_surface} ${distance_out_dir}/left_coords.shape.gii >> ${joblist}
echo wb_command -surface-coordinates-to-metric ${right_surface} ${distance_out_dir}/right_coords.shape.gii >> ${joblist}

echo "" >> ${joblist}

#convert the metric .gii files to a scalar file which has one row for each of x,y,z and one col for each vertex
echo wb_command -cifti-create-dense-scalar ${distance_out_dir}/${subj}.wb_coords.dscalar.nii -right-metric ${distance_out_dir}/right_coords.shape.gii -left-metric ${distance_out_dir}/left_coords.shape.gii >> ${joblist}
echo wb_command -cifti-create-dense-scalar ${distance_out_dir}/${subj}.left_coords.dscalar.nii -left-metric ${distance_out_dir}/left_coords.shape.gii >> ${joblist}
echo wb_command -cifti-create-dense-scalar ${distance_out_dir}/${subj}.right_coords.dscalar.nii -right-metric ${distance_out_dir}/right_coords.shape.gii >> ${joblist}
#convert the scalar file to text - one line for every vertex containing xyz coords
echo wb_command -cifti-convert -to-text ${distance_out_dir}/${subj}.left_coords.dscalar.nii ${distance_out_dir}/${subj}.left_coords.txt >> ${joblist}
echo wb_command -cifti-convert -to-text ${distance_out_dir}/${subj}.right_coords.dscalar.nii ${distance_out_dir}/${subj}.right_coords.txt >> ${joblist}

echo "" >> ${joblist}

#parcellate the scalar coordinates file based on Glasser and weight by vertex area
#this output has 3 maps/rows - one for each of xyz coords, and one col for each parcel in Glasser
echo wb_command -cifti-parcellate ${distance_out_dir}/${subj}.wb_coords.dscalar.nii ${atlas_file} COLUMN ${distance_out_dir}/${subj}.glasser_centroids.pscalar.nii -method MEAN -spatial-weights -left-area-surf ${left_surface} -right-area-surf ${right_surface} >> ${joblist}
#convert to text
echo wb_command -cifti-convert -to-text ${distance_out_dir}/${subj}.glasser_centroids.pscalar.nii ${distance_out_dir}/${subj}.glasser_centroids.txt >> ${joblist}

echo "" >> ${joblist}

#get closest vertex
echo wb_command -surface-closest-vertex ${left_surface} ${distance_out_dir}/${subj}.glasser_centroids.txt ${distance_out_dir}/${subj}.glasser.left_centroid_vnum.txt >> ${joblist}
echo wb_command -surface-closest-vertex ${right_surface} ${distance_out_dir}/${subj}.glasser_centroids.txt ${distance_out_dir}/${subj}.glasser.right_centroid_vnum.txt >> ${joblist}

echo "" >> ${joblist}

#now run python script to create region by region matrix containing euclid distance btwn each region pair
echo python compute_euclid.py ${subj} >> ${joblist}
