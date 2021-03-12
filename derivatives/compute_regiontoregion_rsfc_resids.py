import os, sys
import numpy as np
import pandas as pd
import scipy
from scipy.io import savemat
from sklearn.linear_model import LinearRegression

df=pd.read_csv('../raw_data/merged_sorted_patelmo6_11_24_2020.csv')


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
    return resid #has shape (n_samples,1)

rsfc_resid_t1t2_out = 'rsfc_resid/t1t2/'
if not os.path.exists(rsfc_resid_t1t2_out):
    os.makedirs(rsfc_resid_t1t2_out)
    
rsfc_resid_t1t2_age_sex_out = 'rsfc_resid/t1t2-age-sex/'
if not os.path.exists(rsfc_resid_t1t2_age_sex_out):
    os.makedirs(rsfc_resid_t1t2_age_sex_out)


subject_list=df['Subject'].values 
n_regions=360
n_region_pairs = int((n_regions**2-n_regions)/2)

#compute diff map for each subject
dict_t1t2_diffmap={}
dict_t1t2_diffmap_euclid={}
for subj in subject_list:
    t1t2 = np.loadtxt('../preprocessed/' + str(subj) + '/t1t2/' + str(subj) + '.weighted.parcellated.t1t2.txt')
    #create a diff map - abs val of delta t1t2 btwn all region pairs
    diff_map = np.zeros((n_regions,n_regions))
    #print(np.shape(diff_map))
    for seed in range(0,n_regions):
        for target in range(0,n_regions):
            diff_map[seed,target] = np.abs(t1t2[seed] - t1t2[target])
    dict_t1t2_diffmap[str(subj)]=diff_map.copy()
    
    #now regress out euclidean distance. load the distance data
    euclid_mx = np.loadtxt('../preprocessed/' + str(subj) + '/distance/' + str(subj) + '.euclid_mx.txt')
    euclid_unwrap = unwarp_to_vector(euclid_mx)
    #regress distance against t1t2 diff, store the adjusted diff map for later
    y = unwarp_to_vector(diff_map).reshape(-1,1); x = euclid_unwrap.reshape(-1,1)
    diff_regresseuclid = get_resids(x,y)
    dict_t1t2_diffmap_euclid[str(subj)] = recover_matrix(diff_regresseuclid.flatten(),n_regions).copy()
    
    del t1t2, diff_map, euclid_mx, euclid_unwrap, diff_regresseuclid

#get rsfc for each subject
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
    rsfc_regresseuclid = get_resids(x,y)
    dict_rsfc_euclid[str(subj)] = recover_matrix(rsfc_regresseuclid.flatten(),n_regions).copy()
    del rsfc, euclid_mx, euclid_unwrap,rsfc_regresseuclid

    
#get group avg rsfc matrix
avg_rsfc = sum(dict_rsfc.values())/len(dict_rsfc)
#make matrix containing above threshold connections - thresh is 90% ie top 10% of connections
rsfc_threshold_mx = np.zeros_like(avg_rsfc)
for roi in range(0,np.shape(rsfc_threshold_mx)[0]):
    rsfc_threshold_mx[roi, np.where(avg_rsfc[roi,:] > np.percentile(avg_rsfc[roi,:],90))[0]] = 1
rsfc_threshold_mask = unwarp_to_vector(rsfc_threshold_mx)

fname=rsfc_resid_t1t2_age_sex_out + 'rsfc_threshold_mask.txt'
np.savetxt(fname,rsfc_threshold_mask.astype('float32'),delimiter='\t',fmt='%f')
    
    
#build subject x n_regionpairs matrix for each of diff_map and rsfc
#each col contains data for a given region-region pair
#shapes should be (n_subjects, 64620) forglasser atlas
for subj in subject_list:
    diff_map = dict_t1t2_diffmap[str(subj)] #t1t2
    diff_map_unwrap = unwarp_to_vector(diff_map).reshape(1,n_region_pairs)
    
    rsfc = dict_rsfc[str(subj)] #rsfc
    rsfc_unwrap = unwarp_to_vector(rsfc).reshape(1,n_region_pairs)
    
    if subj==subject_list[0]:
        diff_map_unwrap_mx = diff_map_unwrap.copy()
        rsfc_unwrap_mx = rsfc_unwrap.copy()
    else:
        diff_map_unwrap_mx = np.concatenate((diff_map_unwrap_mx,diff_map_unwrap),axis=0)
        rsfc_unwrap_mx = np.concatenate((rsfc_unwrap_mx,rsfc_unwrap),axis=0)

print(np.shape(diff_map_unwrap_mx), np.shape(rsfc_unwrap_mx))

for subj in subject_list:    
    diff_map = dict_t1t2_diffmap_euclid[str(subj)] #t1t2
    diffmap_unwrap_euclid = unwarp_to_vector(diff_map).reshape(1,n_region_pairs)
    
    rsfc = dict_rsfc_euclid[str(subj)] #rsfc
    rsfc_unwrap_euclid = unwarp_to_vector(rsfc).reshape(1,n_region_pairs)
    
    if subj==subject_list[0]:
        diff_map_unwrap_euclid_mx = diffmap_unwrap_euclid.copy()
        rsfc_unwrap_euclid_mx = rsfc_unwrap_euclid.copy()
    else:
        diff_map_unwrap_euclid_mx = np.concatenate((diff_map_unwrap_euclid_mx,diffmap_unwrap_euclid),axis=0)
        rsfc_unwrap_euclid_mx = np.concatenate((rsfc_unwrap_euclid_mx,rsfc_unwrap_euclid),axis=0)

#now regress for t1t2, age, sex    
age = df['Age_in_Yrs'].values.reshape(-1, 1)
sex_string=df['Gender'].values.reshape(-1, 1)
sex = np.zeros_like(sex_string)
sex[np.where(sex_string=='M'),0]=1
demo_data = np.concatenate((age,sex),axis=1)

age = df['Age_in_Yrs'].values.reshape(-1, 1)
for roiroi in range(0,n_region_pairs):
    x = diff_map_unwrap_euclid_mx[:,roiroi].reshape(-1, 1); y = rsfc_unwrap_euclid_mx[:,roiroi].reshape(-1, 1)
    x = np.concatenate((x,demo_data),axis=1)
    if roiroi == 0:
        rsfc_resid_t1t2_age_sex = get_resids(x,y)
    else:
        rsfc_resid_t1t2_age_sex = np.concatenate((rsfc_resid_t1t2_age_sex, get_resids(x,y)), axis=1)

#now apply mask
valid_idx = np.where(rsfc_threshold_mask > 0)
rsfc_resid_t1t2_age_sex_thresh = rsfc_resid_t1t2_age_sex[:,valid_idx[0]].copy()
    
rsfc_resid_t1t2_age_sex_thresh_shift = rsfc_resid_t1t2_age_sex_thresh - np.min(rsfc_resid_t1t2_age_sex_thresh) #shift for nmf
#save the residualized data to .mat for nmf
#nmf wants region by subjects, so transpose
print("saving", np.shape(np.transpose(rsfc_resid_t1t2_age_sex_thresh_shift)), "nmf with min", np.min(rsfc_resid_t1t2_age_sex_thresh_shift))
savemat(rsfc_resid_t1t2_age_sex_out + "nmf_regionpairsbysubjects.rsfc.residt1t2_age_sex_shift.mat", {"X": np.transpose(rsfc_resid_t1t2_age_sex_thresh_shift)}) 

#save the residualized data to .mat for nmf. save unshifted for stability
#nmf wants region by subjects, so transpose
print("saving", np.shape(np.transpose(rsfc_resid_t1t2_age_sex_thresh)), "nmf with min", np.min(rsfc_resid_t1t2_age_sex_thresh))
savemat(rsfc_resid_t1t2_age_sex_out + "nmf_regionpairsbysubjects.rsfc.residt1t2_age_sex.mat", {"X": np.transpose(rsfc_resid_t1t2_age_sex_thresh)}) 

#write out .txt file per subject
for subj in range(0,np.shape(rsfc_resid_t1t2_age_sex)[0]):
    rsfc_resid_mx = recover_matrix(rsfc_resid_t1t2_age_sex[subj,:], n_regions)
    #write out
    fname=rsfc_resid_t1t2_age_sex_out + str(subject_list[subj]) + '.rsfc_mx.residt1t2_age_sex.txt'
    np.savetxt(fname,rsfc_resid_mx.astype('float32'),delimiter='\t',fmt='%f')
del rsfc_resid_t1t2_age_sex
