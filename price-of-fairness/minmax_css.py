import sys
import datetime
import numpy as np
import reconstruction_error as rc
from numpy.linalg import norm
from time import time
import preprocess_separately as pp
from itertools import combinations

from sys import stdout

def minmax_css(A, B, k):
    """
    Compute the minmax objective value of the column subset selection problem
    min_{S:|S|=k} max{obj(A), obj(B)}

    Parameters
    ----------
    A : array, shape (m, n)
    B : array, shape (m, n)
    k : int

    Returns
    -------
    minmax_obj : optimal objective value for the minmax problem
    min_A : objective value of A, corresponding to the optimal solution
    min_B : objective value of B, corresponding to the optimal solution
    minmax_S : optimal solution for the minmax objective
    """

    m, n = A.shape
    # Compute the rank-k approximation of A and B
    Ak = rc.rank_k_approx(A, k)
    Bk = rc.rank_k_approx(B, k)
    # Compute the norm of the difference between A and Ak, and B and Bk
    A_norm = norm(A - Ak)
    B_norm = norm(B - Bk)

    minmax_obj = np.inf
    minmax_S = []
    min_A = np.inf
    min_B = np.inf

    # Compute the minmax objective value
    for S in list(combinations(range(n), k)):
        CA = A[:, S]
        CB = B[:, S]
        obj_A = rc.normalized_reconstruction_error(A, CA, A_norm)
        obj_B = rc.normalized_reconstruction_error(B, CB, B_norm)
        if max(obj_A, obj_B) < minmax_obj:
            minmax_obj = max(obj_A, obj_B)
            minmax_S = S
            min_A = obj_A
            min_B = obj_B
        #end if
        # Delete variables to save memory
        del CA, CB, obj_A, obj_B
    #end for

    # Delete variables to save memory
    del A, B, Ak, Bk, A_norm, B_norm

    return minmax_obj, min_A, min_B, minmax_S
#end minmax_css()

def minmax_obj(datasets, kk):
    print("###########################################################################")
    print("# Minmax objective")
    # print date and time
    print("# Date: %s"%datetime.datetime.now())
    print("###########################################################################")
    for dataset in datasets:
        dtime = time()
        M, A, B = pp.get_dataset(dataset)
        dtime = time() - dtime

        print("###########################################################################")
        print("# Dataset: %s"%dataset)
        print("# M.shape: %s"%str(M.shape))
        print("# A.shape: %s"%str(A.shape))
        print("# B.shape: %s"%str(B.shape))
        print("# Preprocessing time: %5.2fs"%((dtime)))
        print("###########################################################################")
        print("# k  c  minmax(A,B)  obj(M)  obj(A)   obj(B)  obj(A)-obj(B) totaltime")
        for k in kk:
            totaltime = time()
            minmax_obj, obj_A, obj_B, S = minmax_css(A, B, k)

            CM = M[:, S]
            Mk = rc.rank_k_approx(M, k)
            M_norm = norm(M - Mk)
            obj_M = rc.normalized_reconstruction_error(M, CM, M_norm)
            totaltime = time() - totaltime

            print(" %2d %2d %7.4f %10.4f %7.4f %7.4f %12.4f %7.2f"%\
                  (k, len(S), minmax_obj, obj_M, obj_A, obj_B, obj_A-obj_B,
                   totaltime))
            stdout.flush()

            # Delete variables to save memory
            del CM, Mk, M_norm, obj_M
        #end for k
    #end for dataset
#end minmax_obj()

#datasets = ['heart-cleveland', 'heart-hungarian', 'heart-switzerland', 
#            'heart-va', 'german-credit', 'student-perf']
#datasets = ['student-entrance', 'autism', 'heart-cleveland', 'german-credit',
#            'student-perf']
#datasets = ['autism']
#kk = [3, 4, 5, 6, 7, 8, 9, 10]
#minmax_obj(datasets, kk)

def main():
    dataset = sys.argv[1]
    kk = [3, 4, 5, 6, 7, 8, 9, 10]
    minmax_obj([dataset], kk)
#end main()

if __name__ == "__main__":
    main()
