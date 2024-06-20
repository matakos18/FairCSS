from numpy.linalg import inv, pinv, norm, svd, qr
import numpy as np
import random
import scipy.io
import fair_csslib as fair_css
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
#import preprocessors
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


def split_into_groups(X):
    X= X[:, np.random.permutation(X.shape[1])]
    X= X[np.random.permutation(X.shape[0]), :]
#    X=X[:,:500]
    half=math.floor(m/2)
    A= X[:half, :]
    B= X[half:, :]
    
    return A,B


def compute_algs_over_k(towrite, ls_ranking, A, B, Ak, Bk, f, k_values, c, time_limit=7200):
    f.write(towrite)
    f.write('k,' )
    for k in k_values:
        f.write(str(k)+',')
    f.write('\nc,')
    for k in c:
        f.write(str(c[k])+',')
    f.write('\no,')
    for k in k_values: 
        if 'low qr' in towrite:
            S=fair_css.pivoted_low_qr(A[:,ls_ranking[k]],B[:,ls_ranking[k]],k, pivoting_strategy="max", time_limit=time_limit)
        elif 'high qr' in towrite:
            S=fair_css.pivoted_high_qr(A[:,ls_ranking[k]],B[:,ls_ranking[k]],k, pivoting_strategy="min", time_limit=time_limit)
        elif 'greedy' in towrite:
            S=fair_css.greedy_fixed_denom(A[:,ls_ranking[k]],B[:,ls_ranking[k]], norm(A-Ak[k]), norm(B-Bk[k]), k, time_limit=time_limit)
        if S.size==0:
            f.write('-,')
        else:
            f.write(str(np.round(fair_css.normalized_obj(A,B,ls_ranking[k][S],norm(A-Ak[k]), norm(B-Bk[k])),5))+',')
    f.write('\n')
    f.flush()
    
def compute_obj_over_k(towrite, S, Ak, Bk, f, subset_k_values):
    f.write(towrite)
    f.write('k,' )
    for i in subset_k_values:
        f.write(str(i)+',')
    f.write('\nc,')
    for val in c:
        f.write('-,')
    f.write('\no,')
    for i in subset_k_values:       
        f.write(str(np.round(fair_css.normalized_obj(A,B,S[:i],norm(A-Ak[i]), norm(B-Bk[i])),5))+',')
    f.write('\n')
    f.flush()

    
    
#datasets = ['heart', 'adult', 'german', 'credit', 'student', 'compas', 'communities', 'recidivism', 'mepsh181', 'mepsh192']


# Create the parser
parser = argparse.ArgumentParser(description="obj_over_k")

# Add the arguments
parser.add_argument('dataset', metavar='arg1', type=str, help='the first argument')

# Execute parse_args()
args = parser.parse_args()
dataset=args.dataset

limit=60
starting_k=10

A,B=fair_dataset_loader(dataset)
m=A.shape[0]+B.shape[0]
n=A.shape[1]

print(norm(A))
print(norm(B))

#X=ss.zscore(X)
if not np.isnan(norm(A)) and not np.isnan(norm(B)):
    f = open("normalized_separately/Results_obj_over_k_unit_norm/"+dataset, "w")
    print(dataset)
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


    print(dataset+'(rank A:'+str(np.linalg.matrix_rank(A))+',rank B:'+str(np.linalg.matrix_rank(B))+')\n')
    max_k=min(100,min(int(math.ceil(np.linalg.matrix_rank(A)/2)), int(math.ceil(np.linalg.matrix_rank(B)/2)) ))
    start=starting_k
    #step=max(1,int(math.ceil(max_k/10)))
    values= np.geomspace(start, max_k, num=3)
    k_values = np.unique(np.round(values).astype(int))
    print("k values: ",k_values)


    UA,SA,VtA = svd(A, full_matrices=False)
    UB,SB,VtB = svd(B, full_matrices=False)

    Ak={}
    Bk={}
    ls_ranking={}
    c={}
    for k in k_values:
    #for k in range(start,max_k,step):
        start_time=time.time()
        Ak[k] = UA[:,:k].dot(UA[:,:k].T).dot(A)
        Bk[k] = UB[:,:k].dot(UB[:,:k].T).dot(B)
        #print("ellapsed ", round(time.time()-start_time,6) , "seconds to compute rank k approximations")
        start_time=time.time()
        ls_A=norm(VtA[:k,:],axis=0)**2
        ls_B=norm(VtB[:k,:],axis=0)**2
        df = pd.DataFrame({'ls_A': ls_A, 'ls_B': ls_B})
        df.to_csv("leverage_scores/"+dataset+"_rank_"+str(k), index=False)

        #ls_A,ls_B=fair_css.ls_scores(A,B,k)
        #print("ellapsed ", round(time.time()-start_time,6) , "seconds to compute leverage scores")


#           plt.plot(np.sort(ls_A)[::-1], color='magenta', marker='o',mfc='pink' ) #plot the data
#           plt.plot(np.sort(ls_B)[::-1], color='blue', marker='o',mfc='blue' ) #plot the data

#           plt.show()
        tuples=np.vstack((ls_A,ls_B)).T
        if dataset=='heart' or dataset=='adult' or dataset=='credit' or dataset=='german' or dataset=='student' or dataset=='compas' or dataset=='communities' or dataset=='recidivism':
            theta=k-0.5
        else:
            theta=3.0*k/4.0
        #theta=max(2,k/4.0)
        start_time=time.time()
        c[k],ls_ranking[k]=fair_css.approx_tuple_sampler(tuples,theta,A.shape[1])
        #print("ellapsed ", round(time.time()-start_time,6) , "seconds to sample tuples")
        print("c to achieve theta: ",theta, " is ",c[k])
        curr_num_cols=c[k]
#          while fair_css.normalized_obj(A,B,ls_ranking[k][:curr_num_cols],norm(A-Ak[k]), norm(B-Bk[k]))<=0.5:
#              if curr_num_cols-int(A.shape[1]/100) < int(A.shape[1]/100): 
#                  break
#              curr_num_cols-=int(A.shape[1]/100)
#          print("trimmed to ",curr_num_cols," columns")
#          c[k]=curr_num_cols
#          ls_ranking[k]=ls_ranking[k][:curr_num_cols]

    del UA,UB,SA,SB,VtA,VtB
    

    compute_algs_over_k('#s-low qr\n', ls_ranking, A, B, Ak, Bk, f, k_values, c, time_limit=limit)
    print("finished subsampled low qr")

    
    compute_algs_over_k('#s-high qr\n', ls_ranking, A, B, Ak, Bk, f, k_values, c, time_limit=limit)
    print("finished subsampled high qr")


    compute_algs_over_k('#s-greedy \n', ls_ranking, A, B, Ak, Bk, f, k_values, c, time_limit=limit)
    print("finished subsampled greedy")
    

    S=fair_css.pivoted_low_qr(A,B,max_k, pivoting_strategy="max", time_limit=limit)
    subset_k_values=[i for i in k_values if i<=len(S)]
    compute_obj_over_k('#low qr \n', S, Ak, Bk, f,subset_k_values)
    print("finished low qr")

    
    greedy_ranking=fair_css.greedy(A,B, max_k, time_limit=limit)
    subset_k_values=[i for i in k_values if i<=len(greedy_ranking)]
    compute_obj_over_k('#greedy \n', greedy_ranking, Ak, Bk, f,subset_k_values)
    print("finished greedy")
    
    
    f.write('#random \n')

    holder={}
    random_iterations=50
    print("max k is ", max_k)
    for _ in range(random_iterations):
        random_ranking=np.array(random.sample(range(A.shape[1]),max_k))
        for i in k_values:
        #for i in range(start,max_k,step):
            if i not in holder:
                holder[i]=0
            holder[i]+=fair_css.normalized_obj(A,B,random_ranking[:i],norm(A-Ak[i]), norm(B-Bk[i]))
    f.write('k,')
    for i in holder:
        f.write(str(i)+',')
    f.write('\nc,')
    for i in holder:
        f.write('-,')
    f.write('\no,')
    for i in holder:
        holder[i]=np.round(holder[i]/random_iterations,5)
        f.write(str(holder[i])+',')

    f.close()
