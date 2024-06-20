from numpy.linalg import inv, pinv, norm, svd, qr
import numpy as np
import random
import scipy.io
import fair_csslib as fair_css
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
import preprocess_separately
from tqdm import tqdm
import sys
print(sys.executable)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import scipy.stats as ss
from collections import Counter
import argparse

def read_ssv(dataset_name):
    df=pd.read_csv("Datasets/"+dataset_name, sep=";")
    return df.to_numpy()


def fair_dataset_loader(dataset):
    
    if dataset=='heart':
        A,B=preprocess_separately.preprocess_heart_dataset()
    elif dataset=='adult':
        A,B=preprocess_separately.preprocess_adult_dataset()
    elif dataset=='german':
        A,B=preprocess_separately.preprocess_german_credit_dataset()
    elif dataset=='credit':
        A,B=preprocess_separately.preprocess_credit_card_dataset()
    elif dataset=='student':
        A,B=preprocess_separately.preprocess_student_perf_dataset()
    elif dataset=='compas':
        A,B=preprocess_separately.preprocess_compas_dataset()
    elif dataset=='communities':
        A,B=preprocess_separately.preprocess_communities_dataset()
    elif dataset=='recidivism':
        A,B=preprocess_separately.preprocess_recidivism_juvenile_dataset()
    elif dataset=='mepsh181':
        A,B=preprocess_separately.preprocess_meps_h181_dataset()
    elif dataset=='mepsh192':
        A,B=preprocess_separately.preprocess_meps_h192_dataset()
    return A,B

#datasets = [ 
#    'heart', 'adult', 'german', 'credit', 'student', 'compas', 'communities', 'recidivism', 'mepsh181', 'mepsh192']
#datasets = ['RELATHE.mat']


def compute_obj_over_c(towrite,A,B, S,denomA, denomB, f, c_values):
    f.write(towrite)
    f.write('c,' )
    for i in c_values:
        f.write(str(i)+',')
    f.write('\no,')
    for i in c_values:
        if S.size==0:
            f.write('100,')
        else:
            f.write(str(np.round(fair_css.normalized_obj(A,B,S[:i],denomA, denomB),5))+',')
    f.write('\n')
    f.flush()
    
    
def compute_sampler_over_c(towrite, A, B, denomA, denomB, tuples, f, c_values):
    f.write(towrite)
    f.write('c,' )
    for k in c_values:
        f.write(str(k)+',')
    f.write('\no,')
    for c in c_values:
        _,ls_ranking=fair_css.approx_tuple_sampler(tuples,c,c, time_limit=limit)
        #print("c is: ", c)
        #print("picked this many columns: ", len(ls_ranking))
        #print("picked: ", ls_ranking)
        #print("objective: ", np.round(fair_css.normalized_obj(A,B,ls_ranking,denomA, denomB),5))
        #ls_ranking=sampled_cols[ls_ranking]
        f.write(str(np.round(fair_css.normalized_obj(A,B,ls_ranking,denomA, denomB),5))+',')
    f.write('\n') 
    f.flush()
    
    
    
parser = argparse.ArgumentParser(description="obj_over_c")

# Add the arguments
parser.add_argument('dataset', metavar='arg1', type=str, help='the first argument')

# Execute parse_args()
args = parser.parse_args()
dataset=args.dataset
    
    
    
limit=600
k=50

A,B=fair_dataset_loader(dataset)
n=A.shape[0]+B.shape[0]
m=A.shape[1]

print(norm(A))
print(norm(B))
if not np.isnan(norm(A)) and not np.isnan(norm(B)):
    f = open("normalized_separately/Results_obj_over_c_unit_norm/"+dataset+"_"+str(k), "w")
    print("l2 norm of A: ", norm(A))
    print("l2 norm of B: ", norm(B))
    print("shape of A in ", dataset, "is ", np.shape(A))
    print("shape of B in ", dataset, "is ", np.shape(B))
    
    


    #A,B=split_into_groups(X)

    #while abs(norm(A, ord='fro')-norm(B, ord='fro')) > 0.02*(norm(A, ord='fro')+norm(B, ord='fro')):
    #    A,B=split_into_groups(X)
    #    
    #del X   
    #A=normalize(A)
    #B=normalize(B)
    
    #sampled_cols=fair_css.remove_correlated(A,B,0.8)
    #sampled_VtA = svd(A[:,sampled_cols], full_matrices=False)[2]
    #sampled_VtB = svd(B[:,sampled_cols], full_matrices=False)[2]
    
    #print("sampled this many columns: ",len(sampled_cols))


    print(dataset+'(rank A:'+str(np.linalg.matrix_rank(A))+',rank B:'+str(np.linalg.matrix_rank(B))+')\n')
    #max_c=min(100,min(int(math.ceil(np.linalg.matrix_rank(A)/2)), int(math.ceil(np.linalg.matrix_rank(B)/2)) ))
    #max_c=min(100,min(int(math.ceil(np.linalg.matrix_rank(A))), int(math.ceil(np.linalg.matrix_rank(B))) ))
    min_rank=min(int(math.ceil(np.linalg.matrix_rank(A))), int(math.ceil(np.linalg.matrix_rank(B))) )
    max_c=A.shape[1]-1
    #max_c=len(sampled_cols)
    starting_c=k
    #step=max(1,int(math.ceil(max_k/10)))
    values= np.geomspace(starting_c, max_c, num=10)
    c_values = np.unique(np.round(values).astype(int))
    
    print("c_values: ", c_values)

    UA,SA,VtA = svd(A, full_matrices=False)
    UB,SB,VtB = svd(B, full_matrices=False)



    Ak = UA[:,:k].dot(UA[:,:k].T).dot(A)
    Bk = UB[:,:k].dot(UB[:,:k].T).dot(B)
    denom_A=norm(A-Ak)
    denom_B=norm(B-Bk)

    del UA,UB

    S=fair_css.pivoted_low_qr(A,B,min_rank, pivoting_strategy="max", time_limit=limit)
    compute_obj_over_c('#low qr\n', A,B,S, denom_A, denom_B, f, c_values)


    S=fair_css.pivoted_high_qr(A,B,min_rank, pivoting_strategy="min", time_limit=limit)
    compute_obj_over_c('#high qr\n', A,B,S, denom_A, denom_B, f,c_values)


    #greedy_ranking=fair_css.greedy_fixed_denom(A,B, norm(A-Ak), norm(B-Bk), max_c, time_limit=limit)
    #compute_obj_over_c('#greedy \n', A,B,greedy_ranking, denom_A, denom_B, f,c_values)
    
    

    ls_A=norm(VtA[:k,:],axis=0)**2
    ls_B=norm(VtB[:k,:],axis=0)**2
    tuples=np.vstack((ls_A,ls_B)).T
    #theta=k-(2/3)
    compute_sampler_over_c('#tuple sampler \n', A, B, denom_A, denom_B, tuples, f, c_values)

    print("starting random")
    f.write('#random \n')

    holder={}
    random_iterations=50
    for _ in range(random_iterations):
        random_ranking=np.array(random.sample(range(A.shape[1]),max_c))
        for i in c_values:
            if i not in holder:
                holder[i]=0
            holder[i]+=fair_css.normalized_obj(A,B,random_ranking[:i],denom_A, denom_B)
    f.write('c,')
    for i in holder:
        f.write(str(i)+',')
    f.write('\no,')
    for i in holder:
        holder[i]=np.round(holder[i]/random_iterations,5)
        f.write(str(holder[i])+',')
    f.close()
