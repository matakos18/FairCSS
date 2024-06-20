import numpy as np
from numpy.linalg import pinv, norm, inv, svd, qr
from scipy.linalg import block_diag
import itertools as it
import time
from tqdm import tqdm
from numba import njit
import signal
import pandas as pd
from typing import List, Tuple


def handler(signum, frame):
    raise TimeoutError("Greedy timed out")
@njit
def high_qr_main_loop(i,n,Ra,Rb,P,pivoting_strategy):
    lead_Ra=Ra[:i,:i]
    lead_Rb=Rb[:i,:i]
    Ua,Sa,Vta=svd(lead_Ra, full_matrices=False)
    Ub,Sb,Vtb=svd(lead_Rb, full_matrices=False)

    if pivoting_strategy=="min": 
        if (Sa[-1]>Sb[-1]):
            V=Vtb[-1,:]
        else:
            V=Vta[-1,:]
        #index of pivot element
        index = np.argmax(np.abs(V))
    #        elif pivoting_strategy=="avg":
    #            coeffs=[(0.5*Vta[-1,j]+0.5*Vtb[-1,j]) for j in range(Vta.shape[1])]
    #            index=np.argmax(coeffs)
    #        elif pivoting_strategy=="balance_error":
    #            if norm(lead_Ra, ord='fro') > norm(lead_Rb, ord='fro'):
    #                V=Vtb[-1,:]
    #            else:
    #                V=Vta[-1,:]
    #            #index of pivot element
    #            index = np.argmax(np.abs(V))            

    P_perm=np.identity(i)

    P_perm[np.array([index,i-1])]=P_perm[np.array([i-1,index])]
    #        print("====================")

    Qa_temp,lead_Ra_temp=qr(lead_Ra.dot(P_perm))
    Qb_temp,lead_Rb_temp=qr(lead_Rb.dot(P_perm))
    #        P_temp=block_diag(P_perm,np.identity(n-i))
    P_temp=np.vstack(( np.hstack(( P_perm,np.zeros((i,n-i)) )), np.hstack(( np.zeros((n-i,i)),np.identity(n-i)    )) ))
    P=P.dot(P_temp)
    Ra=lead_Ra_temp
    Rb=lead_Rb_temp

    
    return Ra,Rb,P


def pivoted_high_qr(A,B,k, pivoting_strategy="min", time_limit=7200):
    start_time=time.time()
    n=A.shape[1]
    P=np.identity(n)
    Qa,Ra=qr(A)
    Qb,Rb=qr(B)
    sA=svd(A, full_matrices=False)[1]
    sB=svd(B, full_matrices=False)[1]
    number_columns=0
    for i in range(n,0,-1):
        if (time.time()-start_time)>time_limit:
            #if time limit was exceeded we have no solution at all...
            return np.array([])
        Ra,Rb,P=high_qr_main_loop(i,n,Ra,Rb,P,pivoting_strategy)
        number_columns+=1
        
    S=P.T.dot( np.array(range(A.shape[1])) )
    S=S[:k]
    S=np.array([int(x) for x in S])
    
    return S

@njit
def low_qr_main_loop(i,n,tail_Ra,tail_Rb,P,pivoting_strategy):
    Ua,Sa,Vta=svd(tail_Ra, full_matrices=False)
    Ub,Sb,Vtb=svd(tail_Rb, full_matrices=False)


    if pivoting_strategy=="max":
        if (Sa[0]<Sb[0]):
            V=Vtb[0,:]
        else:
            V=Vta[0,:]
        #index of pivot element
        index = np.argmax(np.abs(V))
#        elif pivoting_strategy=="avg":
#            coeffs=[(0.5*Vta[0,j]+0.5*Vtb[0,j]) for j in range(Vta.shape[1])]
#            index=np.argmax(coeffs)
#        elif pivoting_strategy=="balance_error":
#            if norm(tail_Ra, ord='fro') < norm(tail_Rb, ord='fro'):
#                V=Vtb[0,:]
#            else:
#                V=Vta[0,:]
#            #index of pivot element
#            index = np.argmax(np.abs(V))
    else:
        print("Invalid pivoting strategy")
        return

    P_perm=np.identity(n-i)

    P_perm[np.array([index,0])]=P_perm[np.array([0,index])]
#        print("====================")

    Qa_temp,tail_Ra_temp=qr(tail_Ra.dot(P_perm))
    Qb_temp,tail_Rb_temp=qr(tail_Rb.dot(P_perm))
#        P_temp=block_diag(np.identity(i),P_perm)
    P_temp=np.vstack(( np.hstack(( np.identity(i),np.zeros((i,n-i)) )), np.hstack(( np.zeros((n-i,i)),P_perm ))   )) 
    tail_Ra=tail_Ra_temp[1:,1:]
    tail_Rb=tail_Rb_temp[1:,1:]
    P=P.dot(P_temp)

    
    return tail_Ra,tail_Rb,P


def pivoted_low_qr(A,B,k, pivoting_strategy="max", time_limit=7200):
    start_time=time.time()
    n=A.shape[1]
    P=np.identity(n)
    Qa,tail_Ra=qr(A)
    Qb,tail_Rb=qr(B)
    sA=svd(A, full_matrices=False)[1]
    sB=svd(B, full_matrices=False)[1]
    number_columns=0
    for i in range(k):
        if (time.time()-start_time)>time_limit:
            break
        tail_Ra,tail_Rb,P=low_qr_main_loop(i,n,tail_Ra,tail_Rb,P,pivoting_strategy)
        number_columns+=1
        
    S=P.T.dot( np.array(range(A.shape[1])) )
    S=S[:number_columns]
    S=np.array([int(x) for x in S])
    
    return S
     
    
def approx_tuple_sampler(tuples, threshold, max_limit, verbose=False, time_limit=7200):

    start_time = time.time()
    
    # sort tuples according to their summed values
    I = np.argsort(np.sum(tuples, axis=1))[::-1]

    ## start collecting tuples until one threshold is satisfied
    # current tuple sum
    sums = np.array([0.0, 0.0])
    # counter
    k1 = 0
    # collect original column indices
    selected = []

    while all(sums < threshold):
        if time.time() - start_time > time_limit or k1>=max_limit:
            return k1,np.array(selected)
        sums += tuples[I[k1]]
        selected.append(I[k1])
        k1 += 1

    k2=0
    ## to satisfie the second dimension we just sort the tuples according to their value
    if not all(sums > threshold):

        # dimension which is not satisfied yet
        i = np.argmin(sums)
    
        # take the still available tuples
        rest_tuples = tuples[I[k1:]]
        
        # sort them according to the value in the insatisfied dimension
        J = rest_tuples[:, i].argsort()[::-1]

        while sums[i] < threshold:
            if time.time() - start_time > time_limit or k1+k2 >=max_limit:
                return k1+k2,np.array(selected)
            sums[i] += rest_tuples[J[k2]][i]

            # get original index
            o = I[k1 + J[k2]]

            selected.append(o)
            k2 += 1

    k_total=k1+k2

    return k_total, np.array(selected)     



def tuple_sampler(tuples, threshold, max_limit, verbose=False, time_limit=7200):
    start_time=time.time()
    not_selected_tuples=tuples
    not_selected_inds=np.arange(len(tuples))
    selected=[]
    k=0
    curr_sum=[0,0]
    while k<max_limit and (time.time()-start_time)<time_limit:
        curr_dist_0=threshold-curr_sum[0]
        curr_dist_1=threshold-curr_sum[1]
        contribs=[min(tple[0],curr_dist_0)+min(tple[1],curr_dist_1) for tple in not_selected_tuples]
        index_max_contrib=np.argmax(contribs)
        curr_sum+=tuples[not_selected_inds[index_max_contrib]]
        selected.append(not_selected_inds[index_max_contrib])
        not_selected_tuples = np.delete(not_selected_tuples, index_max_contrib, axis=0)
        not_selected_inds = np.delete(not_selected_inds,index_max_contrib, axis=0)
        k+=1
        if min(curr_sum[0],curr_sum[1])>=threshold:
            return k,selected

    return k,selected


def obj_common_pinv(X,A,B,S, verbose=False):
    CA = A[:,S]
    CB = B[:,S]
    C = X[:,S]
    pinvC=pinv(C)
    objA=norm(A-CA.dot(pinvC.dot(X)))
    objB=norm(B-CB.dot(pinvC.dot(X)))
    if verbose:
        print("Error for group A: ",objA)
        print("Error for group B: ",objB)
    return max(objA, objB)

def obj(X,A,B,S, verbose=False):
    CA = A[:,S]
    CB = B[:,S]
    C = X[:,S]
    pinvCA=pinv(CA)
    pinvCB=pinv(CB)
    objA=norm(A-CA.dot(pinvCA.dot(A)), ord='fro')#/norm(A, ord='fro')**2
    objB=norm(B-CB.dot(pinvCB.dot(B)), ord='fro')#/norm(B, ord='fro')**2
    if verbose:
        print("Error for group A: ",objA)
        print("Error for group B: ",objB)
    return max(objA, objB)

@njit
def compute_obj(A,B,CA,CB, A_normalization, B_normalization):
    pinvCA=pinv(CA)
    pinvCB=pinv(CB)
    proj_coeffsA=pinvCA.dot(A)
    proj_coeffsB=pinvCB.dot(B)
    objA=norm(A-CA.dot(proj_coeffsA))/A_normalization
    objB=norm(B-CB.dot(proj_coeffsB))/B_normalization
    return objA,objB

@njit
def normalized_obj(A,B,S, A_normalization, B_normalization, verbose=False):
    CA = A[:,S]
    CB = B[:,S]
    objA,objB=compute_obj(A,B,CA,CB, A_normalization, B_normalization)
    if verbose:
        print("Error for group A: ",objA)
        print("Error for group B: ",objB)
    return max(objA, objB)

def proj(A):
    return A.dot(pinv(A))

@njit
def greedy_step(A,B,UAk,UBk,S,not_selected):
    best = 0
    low_dim_A=UAk.T.dot(A)
    Ak = UAk.dot(low_dim_A)
    low_dim_B=UBk.T.dot(B)
    Bk = UBk.dot(low_dim_B)
    denomA=norm(A-Ak)
    denomB=norm(B-Bk)
    minv= normalized_obj(A,B,S,denomA, denomB)
    for candidate in range(A.shape[1]):
        if candidate in not_selected:
            Sf = S.copy()
            Sf=np.append(Sf,candidate)
            val = normalized_obj(A,B, Sf,denomA, denomB)
            if val <= minv:
                best = candidate
                minv = val
    return best

def greedy(A,B, stopping, time_limit=7200):
    S = np.array([],dtype=int)
    i=0
    not_selected=set(range(A.shape[1]))
    UA = svd(A)[0]
    UB = svd(B)[0]
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_limit)
    while i<stopping:
        UAk=UA[:,:i]
        UBk=UB[:,:i]
        try:
            best=greedy_step(A,B,UAk,UBk,S,not_selected)
        except TimeoutError as e:
            print("run out of time")
            return S
        S=np.append(S,best)
        not_selected.remove(best)
        i+=1
    signal.alarm(0)
    return S

@njit
def greedy_fixed_denom_step(A,B,denomA,denomB,S,not_selected):
    best = 0
    minv= round(normalized_obj(A,B,S,denomA, denomB),5)
    for candidate in range(A.shape[1]):
        if candidate in not_selected:
            Sf = S.copy()
            Sf=np.append(Sf,candidate)
            val = round(normalized_obj(A,B, Sf,denomA, denomB),5)
            if val <= minv+0.001:
                best = candidate
                minv = val
    return best    


def greedy_fixed_denom(A,B, denomA, denomB, stopping, time_limit=7200):
    start_time=time.time()
    S = np.array([],dtype=int)
    i=0
    not_selected=set(range(A.shape[1]))
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_limit)
    while i<stopping:
        try:
            best=greedy_fixed_denom_step(A,B,denomA,denomB,S,not_selected)
        except TimeoutError as e:
            print("run out of time")
            return S
        S=np.append(S,best)
        not_selected.remove(best)
        i+=1
    signal.alarm(0)
    return S

def ls_scores(A,B,k):
    # Find the singular vectors most "correlated" with the target vector (pick the set R)
    UA,SA,VtA = svd(A)
    UB,SB,VtB = svd(B)

    #l_scores = np.sort(norm(Vt[R,:],axis=0)**2)[::-1]
    start = time.time()
    ls_A=norm(VtA[:k,:],axis=0)**2
    ls_B=norm(VtB[:k,:],axis=0)**2
    
#    l_rank_A = np.argsort(ls_A)[::-1]
#    l_rank_B = np.argsort(ls_A)[::-1]
    
    return ls_A, ls_B


def combs(ls_A, ls_B, thresh, k):
    not_selected=list(range(len(ls_A)))
    i=1
    while i<=k:
        print(i)
        all_combs=list(it.combinations(not_selected, i))
        for comb in all_combs:
            comb=list(comb)
            if (sum(ls_A[comb])>= thresh) and (sum(ls_B[comb]) >= thresh):
                return comb
        i+=1
        delete(all_combs)


    print("No matching combination for given k")
    return []
        
    

def find_minScores(ls_A, ls_B, thresh):
    selected=[]
    selected=minScores(not_selected, ls_A, ls_B, selected, thresh)
    return selected

 
def minScores(not_selected, ls_A, ls_B, selected, thresh):
   
    for i in not_selected:
        if (sum(ls_A[selected])+ls_A[i] >= thresh) and (sum(ls_B[selected])+ls_B[i] >= thresh):
            selected.append(i)
            return selected
            
    for i in not_selected:
            new_not_selected=not_selected.copy()
            new_not_selected.remove(i)
            new_selected=selected.copy()
            new_selected.append(i)
            solution=minScores(new_not_selected, ls_A, ls_B, new_selected, thresh)
        #if we did not recurse that means that we have satisfied both thresholds

    return solution


def remove_correlated(A,B, threshold):
    
    correlated_columns_A = find_correlated_columns(A, threshold)
    correlated_columns_B = find_correlated_columns(B, threshold)
    
    # Merge the two lists and keep only unique entries
    cols = np.array(list(set(correlated_columns_A + correlated_columns_B)))

    return cols
    
    
    
    
    
def find_correlated_columns(C: np.ndarray, threshold: float) -> List[int]:
    # Compute the correlation matrix
    #corr_matrix = np.corrcoef(C, rowvar=False)
    corr_matrix = C.T@C

    # Convert it into a pandas DataFrame for easier handling
    corr_df = pd.DataFrame(corr_matrix)

    # Create an empty list to store the indices of columns with largest norm in each group
    columns_with_largest_norm = []

    # Find groups of columns with correlation above the threshold
    for i in range(corr_df.shape[0]):
        # Get list of columns where correlation is above the threshold
        correlated_cols = list(corr_df.index[corr_df[i] > threshold])
        #print("this many correlated columns: ",len(correlated_cols))

        # Skip if there is only one column in the group
        if len(correlated_cols) == 1:
            columns_with_largest_norm.append(i)
            continue
        elif len(correlated_cols) <1:
            continue

        # Compute the product of every column with the matrix formed of all other columns
        inner_products = []
        for col in correlated_cols:
            other_cols = [c for c in correlated_cols if c != col]
            product = C[:, col] @ C[:, other_cols]
            norm = np.linalg.norm(product)
            inner_products.append(norm)

        # Find the column with the largest norm
        column_with_largest_norm = correlated_cols[np.argmax(inner_products)]
        columns_with_largest_norm.append(column_with_largest_norm)

    return columns_with_largest_norm
    

