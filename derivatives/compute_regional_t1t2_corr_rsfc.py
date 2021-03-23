import os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy
from scipy.io import savemat

df=pd.read_csv('../raw_data/merged_sorted_r277_unrelated_setA_n384.csv')

subject_list=df['Subject'].values 

t1t2_out_dir = 't1t2/'
if not os.path.exists(t1t2_out_dir):
    os.makedirs(t1t2_out_dir)
    
rsfc_out_dir = 'rsfc/'
if not os.path.exists(rsfc_out_dir):
    os.makedirs(rsfc_out_dir)
    
t1t2corrrsfc_out_dir = 't1t2_corr_rsfc/'
if not os.path.exists(t1t2corrrsfc_out_dir):
    os.makedirs(t1t2corrrsfc_out_dir)

dict_t1t2_diffmap={}
n_regions=360

#compute diff map for each subject
for subj in subject_list:
    t1t2 = np.loadtxt('../preprocessed/' + str(subj) + '/t1t2/' + str(subj) + '.weighted.parcellated.t1t2.txt')
    #create a diff map - abs val of delta t1t2 btwn all region pairs
    diff_map = np.zeros((n_regions,n_regions))
    print(np.shape(diff_map))
    for seed in range(0,n_regions):
        for target in range(0,n_regions):
            diff_map[seed,target] = np.abs(t1t2[seed] - t1t2[target])
    dict_t1t2_diffmap[str(subj)]=diff_map.copy()
    #save a copy of the diff map for each subject
    fname = t1t2_out_dir + str(subj) + '.t1t2.absdiffmap.txt'
    np.savetxt(fname,diff_map.astype('float32'),delimiter='\t',fmt='%f')
    del t1t2, diff_map

#compute group avg
n_subjects = len(dict_t1t2_diffmap)
groupavg_diffmap = dict_t1t2_diffmap[str(subject_list[0])].copy()
for subj in subject_list[1:]:
    groupavg_diffmap = np.add(groupavg_diffmap, dict_t1t2_diffmap[str(subj)])
groupavg_diffmap=np.divide(groupavg_diffmap, n_subjects)
fname = t1t2_out_dir + 'groupavg.t1t2.absdiffmap.txt'
np.savetxt(fname,groupavg_diffmap.astype('float32'),delimiter='\t',fmt='%f')

plt.figure(figsize=(2.5,2),dpi=200)
plt.imshow(groupavg_diffmap,vmin=0,vmax=0.5)
plt.title('Group Avg T1T2\nDelta',fontsize=10)
plt.colorbar()
plt.savefig(t1t2_out_dir + "groupavg.t1t2.absdiffmap.png",bbox_inches='tight', dpi = 'figure')

#load rsfc for each subject
dict_rsfc = {}
for subj in subject_list:
    rsfc=np.loadtxt('../preprocessed/' + str(subj) + '/rsfc/' + str(subj) + '.rfMRI_REST1.netmat.txt')
    np.fill_diagonal(rsfc,0)
    dict_rsfc[str(subj)]=rsfc.copy()
    del rsfc

#compute group avg
n_subjects = len(dict_rsfc)
groupavg_rsfc = dict_rsfc[str(subject_list[0])].copy()
for subj in subject_list[1:]:
    groupavg_rsfc = np.add(groupavg_rsfc, dict_rsfc[str(subj)])
groupavg_rsfc=np.divide(groupavg_rsfc, n_subjects)
fname = rsfc_out_dir + 'groupavg.rsfc.txt'
np.savetxt(fname,groupavg_rsfc.astype('float32'),delimiter='\t',fmt='%f')

plt.figure(figsize=(2.5,2),dpi=200)
plt.imshow(groupavg_rsfc,vmin=-0.25,vmax=0.75)
plt.title('Group Avg RSFC',fontsize=10)
plt.colorbar()
plt.savefig(rsfc_out_dir + "groupavg.rsfc.png",bbox_inches='tight', dpi = 'figure')

#compute corr btwn deltaT1T2 and RSFC
subj=100206
template_txt = np.loadtxt('../preprocessed/' + str(subj) + '/t1t2/' + str(subj) + '.weighted.parcellated.t1t2.txt')
t1t2_rsfc_corr = np.zeros_like(template_txt)
np.fill_diagonal(groupavg_rsfc,0.5)
for roi in range(0,n_regions):
    t1t2_rsfc_corr[roi] = np.corrcoef(groupavg_rsfc[roi,:], groupavg_diffmap[roi,:])[0,1]

np.savetxt(t1t2corrrsfc_out_dir + 'groupavg.regional.t1t2_corr_rsfc.txt',t1t2_rsfc_corr.astype('float32'),delimiter='\t',fmt='%f')

#compute micro-func relationship across subjects

for subj in subject_list:
    diff_map = dict_t1t2_diffmap[str(subj)] #t1t2
    
    rsfc = dict_rsfc[str(subj)] #rsfc

    t1t2_rsfc_corr = np.zeros_like(template_txt)
    for roi in range(0,n_regions):
        t1t2_rsfc_corr[roi] = np.corrcoef(diff_map[roi,:], rsfc[roi,:])[0,1]
    np.savetxt(t1t2corrrsfc_out_dir + str(subj) + '.regional.t1t2_corr_rsfc.txt',t1t2_rsfc_corr.astype('float32'),delimiter='\t',fmt='%f')
    
    if subj==subject_list[0]:
        group_t1t2_rsfc_corr = t1t2_rsfc_corr.reshape(n_regions,1)
    else:
        group_t1t2_rsfc_corr = np.concatenate((group_t1t2_rsfc_corr,t1t2_rsfc_corr.reshape(n_regions,1)),axis=1)

groupmean_t1t2_rsfc_corr = np.mean(group_t1t2_rsfc_corr,axis=1)
np.savetxt(t1t2corrrsfc_out_dir + 'groupmean.regional.t1t2_corr_rsfc.txt',groupmean_t1t2_rsfc_corr.astype('float32'),delimiter='\t',fmt='%f')

groupstd_t1t2_rsfc_corr = np.std(group_t1t2_rsfc_corr,axis=1)
np.savetxt(t1t2corrrsfc_out_dir + 'groupstd.regional.t1t2_corr_rsfc.txt',groupstd_t1t2_rsfc_corr.astype('float32'),delimiter='\t',fmt='%f')

grouprelstd_t1t2_rsfc_corr = np.abs(np.divide(groupstd_t1t2_rsfc_corr,groupmean_t1t2_rsfc_corr))
np.savetxt(t1t2corrrsfc_out_dir + 'grouprelativestd.regional.t1t2_corr_rsfc.txt',grouprelstd_t1t2_rsfc_corr.astype('float32'),delimiter='\t',fmt='%f')
    
#group_t1t2_rsfc_corr is n_regions x n_subjects but will have neg values
#shift it to make positive, then save

nmf_group_t1t2_rsfc_corr_shifted = group_t1t2_rsfc_corr - np.min(group_t1t2_rsfc_corr)
np.savetxt(t1t2corrrsfc_out_dir + 'nmf_regionxsubjects.shifted.t1t2_corr_rsfc.txt',nmf_group_t1t2_rsfc_corr_shifted.astype('float32'),delimiter='\t',fmt='%f')

#invert...the corrs that are -ve in this case are maybe more interesting. so *-1, then shift
group_t1t2_rsfc_corr_flipped = np.multiply(group_t1t2_rsfc_corr,-1)
nmf_group_t1t2_rsfc_corr_flipped_shifted = group_t1t2_rsfc_corr_flipped - np.min(group_t1t2_rsfc_corr_flipped)
np.savetxt(t1t2corrrsfc_out_dir + 'nmf_regionxsubjects.flipped.shifted.t1t2_corr_rsfc.txt',nmf_group_t1t2_rsfc_corr_flipped_shifted.astype('float32'),delimiter='\t',fmt='%f')
    
#save a version thats just the abs val as well    
nmf_group_t1t2_rsfc_corr_abs = np.abs(group_t1t2_rsfc_corr)
np.savetxt(t1t2corrrsfc_out_dir + 'nmf_regionxsubjects.abs.t1t2_corr_rsfc.txt',nmf_group_t1t2_rsfc_corr_abs.astype('float32'),delimiter='\t',fmt='%f')

savemat(t1t2corrrsfc_out_dir + "nmf_regionxsubjects.shifted.t1t2_corr_rsfc.mat", {"X": nmf_group_t1t2_rsfc_corr_shifted})
savemat(t1t2corrrsfc_out_dir + "nmf_regionxsubjects.shifted.t1t2_corr_rsfc.mat", {"X": nmf_group_t1t2_rsfc_corr_shifted})
savemat(t1t2corrrsfc_out_dir + "nmf_regionxsubjects.flipped.shifted.t1t2_corr_rsfc.mat", {"X": nmf_group_t1t2_rsfc_corr_flipped_shifted})
    
    
    
    
    


