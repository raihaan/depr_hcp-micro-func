import os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy
from scipy.io import savemat

df=pd.read_csv('../raw_data/merged_sorted_r277_unrelated_setA_n384.csv')

def cumul(n): #thanks Gabe DG
    if n == 0:
        return 0
    else:
        return n + cumul(n-1)

def unwarp_to_vector(matrix):
    vector=np.empty([int((len(matrix)**2-len(matrix))/2)])
    vector[0:1]=matrix[1,:1]
    for i in range(1,len(matrix)):
        vector[cumul(i-1):cumul(i)]=matrix[i,:i]
    return vector

def recover_matrix(FC_vector,num_roi):
    matrix=np.zeros([num_roi, num_roi])
    for i in range(1,num_roi):
        matrix[i,:i]=FC_vector[cumul(i-1):cumul(i)]
        matrix[:i,i]=FC_vector[cumul(i-1):cumul(i)]
    np.fill_diagonal(matrix, 0)
    return matrix

def get_resids(x, y):
    regr = LinearRegression().fit(x,y)
    resid = y - regr.predict(x)
    return resid.reshape(-1,1), regr.coef_ #has shape (n_samples,1)


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
dict_t1t2_diffmap_euclid={}
n_regions=360

#compute diff map for each subject
for subj in subject_list:
    t1t2 = np.loadtxt('../preprocessed/' + str(subj) + '/t1t2/' + str(subj) + '.weighted.parcellated_bc.t1t2.txt')
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

    #now regress out euclidean distance. load the distance data
    euclid_mx = np.loadtxt('../preprocessed/' + str(subj) + '/distance/' + str(subj) + '.euclid_mx.txt')
    euclid_unwrap = unwarp_to_vector(euclid_mx)
    #regress distance against t1t2 diff, store the adjusted diff map for later
    y = unwarp_to_vector(diff_map).reshape(-1,1); x = euclid_unwrap.reshape(-1,1)
    diff_regresseuclid, coeff = get_resids(x,y)
    dict_t1t2_diffmap_euclid[str(subj)] = recover_matrix(diff_regresseuclid.flatten(),n_regions).copy()
    del t1t2, diff_map

#compute group avg
n_subjects = len(dict_t1t2_diffmap)
groupavg_diffmap = dict_t1t2_diffmap[str(subject_list[0])].copy()
for subj in subject_list[1:]:
    groupavg_diffmap = np.add(groupavg_diffmap, dict_t1t2_diffmap[str(subj)])
groupavg_diffmap=np.divide(groupavg_diffmap, n_subjects)
fname = t1t2_out_dir + 'groupavg.t1t2.absdiffmap.txt'
np.savetxt(fname,groupavg_diffmap.astype('float32'),delimiter='\t',fmt='%f')

#repeat for distance corrected
del groupavg_diffmap
groupavg_diffmap = dict_t1t2_diffmap_euclid[str(subject_list[0])].copy()
for subj in subject_list[1:]:
    groupavg_diffmap = np.add(groupavg_diffmap, dict_t1t2_diffmap_euclid[str(subj)])
groupavg_diffmap=np.divide(groupavg_diffmap, n_subjects)
fname = t1t2_out_dir + 'groupavg.t1t2_euclid.absdiffmap.txt'
np.savetxt(fname,groupavg_diffmap.astype('float32'),delimiter='\t',fmt='%f')

#load rsfc for each subject
dict_rsfc = {}
dict_rsfc_euclid = {}
for subj in subject_list:
    rsfc=np.loadtxt('../preprocessed/' + str(subj) + '/rsfc/' + str(subj) + '.rfMRI_REST1.netmat.txt')
    np.fill_diagonal(rsfc,0)
    dict_rsfc[str(subj)]=rsfc.copy()

    #now regress out euclidean distance. load the distance data
    euclid_mx = np.loadtxt('../preprocessed/' + str(subj) + '/distance/' + str(subj) + '.euclid_mx.txt')
    euclid_unwrap = unwarp_to_vector(euclid_mx)
    #regress distance against t1t2 diff, store the adjusted diff map for later
    y = unwarp_to_vector(rsfc).reshape(-1,1); x = euclid_unwrap.reshape(-1,1)
    rsfc_regresseuclid, coeff = get_resids(x,y)
    dict_rsfc_euclid[str(subj)] = recover_matrix(rsfc_regresseuclid.flatten(),n_regions).copy()
    del rsfc

#compute group avg
groupavg_rsfc = dict_rsfc[str(subject_list[0])].copy()
for subj in subject_list[1:]:
    groupavg_rsfc = np.add(groupavg_rsfc, dict_rsfc[str(subj)])
groupavg_rsfc=np.divide(groupavg_rsfc, n_subjects)
fname = rsfc_out_dir + 'groupavg.rsfc.txt'
np.savetxt(fname,groupavg_rsfc.astype('float32'),delimiter='\t',fmt='%f')
del groupavg_rsfc

#repeat for distance corrected rsfct
groupavg_rsfc = dict_rsfc_euclid[str(subject_list[0])].copy()
for subj in subject_list[1:]:
    groupavg_rsfc = np.add(groupavg_rsfc, dict_rsfc_euclid[str(subj)])
groupavg_rsfc=np.divide(groupavg_rsfc, n_subjects)
fname = rsfc_out_dir + 'groupavg.rsfc_euclid.txt'
np.savetxt(fname,groupavg_rsfc.astype('float32'),delimiter='\t',fmt='%f')



#compute corr btwn deltaT1T2 and RSFC
subj=subject_list[0]
template_txt = np.loadtxt('../preprocessed/' + str(subj) + '/t1t2/' + str(subj) + '.weighted.parcellated_bc.t1t2.txt')
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
    
