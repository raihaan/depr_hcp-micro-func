import os, sys
import numpy as np
import pandas as pd
import scipy
from scipy.io import savemat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

def measure_distance_effect(raw,corrected,distance):
    metrics_list=[]
    metrics_list.append(np.corrcoef(raw,corrected)[0,1]) #correlate raw and corrected
    metrics_list.append(np.corrcoef(raw,distance)[0,1]) #correlate raw and distance
    metrics_list.append(np.corrcoef(corrected,distance)[0,1]) #correlate corrected distance
    metrics_list.append(mean_squared_error(raw, corrected)) #mean sq error
    return metrics_list

rsfc_resid_t1t2_age_sex_out = 'rsfc_resid/t1t2-age-sex/'
if not os.path.exists(rsfc_resid_t1t2_age_sex_out):
    os.makedirs(rsfc_resid_t1t2_age_sex_out)

rsfc_resid_t1t2_out = 'rsfc_resid/t1t2/'
if not os.path.exists(rsfc_resid_t1t2_out):
    os.makedirs(rsfc_resid_t1t2_out)

subject_list=df['Subject'].values 
n_regions=360
n_region_pairs = int((n_regions**2-n_regions)/2)

#compute diff map for each subject
dict_t1t2_diffmap={}
dict_t1t2_diffmap_euclid={}
dict_t1t2_distance_effects={}
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
    diff_regresseuclid, coeff = get_resids(x,y)
    dict_t1t2_diffmap_euclid[str(subj)] = recover_matrix(diff_regresseuclid.flatten(),n_regions).copy()

    #get distance effects on raw and corrected data
    dict_t1t2_distance_effects[str(subj)] = measure_distance_effect(
        raw = unwarp_to_vector(dict_t1t2_diffmap[str(subj)]),
        corrected = unwarp_to_vector(dict_t1t2_diffmap_euclid[str(subj)]),
        distance = euclid_unwrap)
    
    del t1t2, diff_map, euclid_mx, euclid_unwrap, diff_regresseuclid

#create dataframe containing one row for each subject, one col for each t1t2~distance metric
t1t2_distance_df=pd.DataFrame.from_dict(
    dict_t1t2_distance_effects,orient='index',columns=[
        'T1T2_corr_T1T2corrected','T1T2_corr_euclid','T1T2corrected_corr_euclid','T1T2_T1T2corrected_mse'])
t1t2_distance_df.index.name='Subject'

#get rsfc for each subject
dict_rsfc = {}
dict_rsfc_euclid = {}
dict_rsfc_distance_effects={}
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

    #get distance effects on raw and corrected data
    dict_rsfc_distance_effects[str(subj)] = measure_distance_effect(
        raw = unwarp_to_vector(dict_rsfc[str(subj)]),
        corrected = unwarp_to_vector(dict_rsfc_euclid[str(subj)]),
        distance = euclid_unwrap)
    del rsfc, euclid_mx, euclid_unwrap,rsfc_regresseuclid

#create dataframe containing one row for each subject, one col for each t1t2~distance metric
rsfc_distance_df=pd.DataFrame.from_dict(
    dict_rsfc_distance_effects,orient='index',columns=[
        'RSFC_corr_RSFCcorrected','RSFC_corr_euclid','RSFCcorrected_corr_euclid','RSFC_RSFCcorrected_mse'])
rsfc_distance_df.index.name='Subject'

#merge and write out t1t2 and rsfc distance dfs
distance_df = t1t2_distance_df.merge(rsfc_distance_df, how='inner', on='Subject')
distance_df.round(5).to_csv('distance_metrics.csv') #round to 5 sig digits for output

    
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

fname=rsfc_resid_t1t2_age_sex_out + 'diffmap_unwrap.txt'
np.savetxt(fname,diff_map_unwrap_mx.astype('float32'),delimiter='\t',fmt='%f')
fname=rsfc_resid_t1t2_age_sex_out + 'rsfc_unwrap.txt'
np.savetxt(fname,rsfc_unwrap_mx.astype('float32'),delimiter='\t',fmt='%f')

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

fname=rsfc_resid_t1t2_age_sex_out + 'diffmap_unwrap_euclid.txt'
np.savetxt(fname,diff_map_unwrap_euclid_mx.astype('float32'),delimiter='\t',fmt='%f')
fname=rsfc_resid_t1t2_age_sex_out + 'rsfc_unwrap_euclid.txt'
np.savetxt(fname,rsfc_unwrap_euclid_mx.astype('float32'),delimiter='\t',fmt='%f')

#now regress for delta t1t2
#now regress for t1t2 
betas_vector = np.zeros((n_region_pairs,1)) #array for tracking coefficient of delta t1t2 in regression on each region pair
for roiroi in range(0,n_region_pairs):
    #get rsfc and delta t1t2 data
    x = diff_map_unwrap_euclid_mx[:,roiroi].reshape(-1, 1); y = rsfc_unwrap_euclid_mx[:,roiroi]
    if roiroi == 0:
        rsfc_resid_t1t2, coeff = get_resids(x,y) #compute residuals and get model coefficient 
    else:
        rsfc_resids, coeff = get_resids(x,y)
        rsfc_resid_t1t2 = np.concatenate((rsfc_resid_t1t2, rsfc_resids), axis=1)
    betas_vector[roiroi,0] = coeff[0] #store coefficient

#save raw residuals
fname=rsfc_resid_t1t2_out + 'raw_subjectsbyregionpairs.rsfc.residt1t2.txt'
np.savetxt(fname,rsfc_resid_t1t2.astype('float32'),delimiter='\t',fmt='%f')

#recast beta vector to matrix (n_regions x n_regions) form and save
betas_mx = recover_matrix(betas_vector.flatten(),n_regions) 
fname=rsfc_resid_t1t2_out + 't1t2_betas_mx.txt'
np.savetxt(fname,betas_mx.astype('float32'),delimiter='\t',fmt='%f')

#now apply mask
valid_idx = np.where(rsfc_threshold_mask > 0)
rsfc_resid_t1t2_thresh = rsfc_resid_t1t2[:,valid_idx[0]].copy()
    
rsfc_resid_t1t2_thresh_shift = rsfc_resid_t1t2_thresh - np.min(rsfc_resid_t1t2_thresh) #shift for nmf
#save the residualized data to .mat for nmf
#nmf wants region by subjects, so transpose
print("saving", np.shape(np.transpose(rsfc_resid_t1t2_thresh_shift)), "nmf with min", np.min(rsfc_resid_t1t2_thresh_shift))
savemat(rsfc_resid_t1t2_out + "nmf_regionpairsbysubjects.rsfc.residt1t2_shift.mat", {"X": np.transpose(rsfc_resid_t1t2_thresh_shift)}) 

#save the residualized data to .mat for nmf. save unshifted for stability
#nmf wants region by subjects, so transpose
print("saving", np.shape(np.transpose(rsfc_resid_t1t2_thresh)), "nmf with min", np.min(rsfc_resid_t1t2_thresh))
savemat(rsfc_resid_t1t2_out + "nmf_regionpairsbysubjects.rsfc.residt1t2.mat", {"X": np.transpose(rsfc_resid_t1t2_thresh)}) 

#write out .txt file per subject
for subj in range(0,np.shape(rsfc_resid_t1t2)[0]):
    rsfc_resid_mx = recover_matrix(rsfc_resid_t1t2[subj,:], n_regions)
    #write out
    fname=rsfc_resid_t1t2_out + str(subject_list[subj]) + '.rsfc_mx.residt1t2.txt'
    np.savetxt(fname,rsfc_resid_mx.astype('float32'),delimiter='\t',fmt='%f')
del rsfc_resid_t1t2, betas_vector, betas_mx, rsfc_resid_t1t2_thresh, rsfc_resid_t1t2_thresh_shift





#now regress for delta t1t2, age, sex    
age = df['Age_in_Yrs'].values.reshape(-1, 1)
sex_string=df['Gender'].values.reshape(-1, 1)
sex = np.zeros_like(sex_string)
sex[np.where(sex_string=='M'),0]=1
demo_data = np.concatenate((age,sex),axis=1)

age = df['Age_in_Yrs'].values.reshape(-1, 1)
betas_vector = np.zeros((n_region_pairs,1)) #array for tracking coefficient of delta t1t2 in regression on each region pair
for roiroi in range(0,n_region_pairs):
    #get rsfc and delta t1t2 data
    x = diff_map_unwrap_euclid_mx[:,roiroi].reshape(-1, 1); y = rsfc_unwrap_euclid_mx[:,roiroi]
    x = np.concatenate((x,demo_data),axis=1) #append demographic (Age,sex) data
    if roiroi == 0:
        rsfc_resid_t1t2_age_sex, coeff = get_resids(x,y) #compute residuals and get model coefficient 
    else:
        rsfc_resids, coeff = get_resids(x,y)
        rsfc_resid_t1t2_age_sex = np.concatenate((rsfc_resid_t1t2_age_sex, rsfc_resids), axis=1)
    betas_vector[roiroi,0] = coeff[0] #store coefficient

#save raw residuals
fname=rsfc_resid_t1t2_age_sex_out + 'raw_subjectsbyregionpairs.rsfc.residt1t2_age_sex.txt'
np.savetxt(fname,rsfc_resid_t1t2_age_sex.astype('float32'),delimiter='\t',fmt='%f')

#recast beta vector to matrix (n_regions x n_regions) form and save
betas_mx = recover_matrix(betas_vector.flatten(),360) 
fname=rsfc_resid_t1t2_age_sex_out + 't1t2_betas_resid_age_sex_mx.txt'
np.savetxt(fname,betas_mx.astype('float32'),delimiter='\t',fmt='%f')

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

