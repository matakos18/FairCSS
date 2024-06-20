import sys
import datetime
import numpy as np
from time import time
from numpy.linalg import norm
from itertools import combinations
import preprocess_separately as pp
import reconstruction_error as rc
from sys import stdout

###############################################################################
def vanilla_css(M, A, B, k):
    """
    Computes the optimal objective value for the problem of finding the best
    k-dimensional subspace for M, A, and B.

    Parameters:
        M (np.ndarray): Input matrix of shape (m, n).
        A (np.ndarray): Input matrix of shape (m, n).
        B (np.ndarray): Input matrix of shape (m, n).
        k (int): Desired dimension of the subspace.

    Returns:
        min_obj: Optimal objective value for the problem of finding the best
        k-dimensional subspace for M
        obj_A: Optimal objective value for the problem of finding the best
        k-dimensional subspace for A
        obj_B: Optimal objective value for the problem of finding the best
        k-dimensional subspace for B
    """

    m, n = M.shape
    Mk = rc.rank_k_approx(M, k)
    Ak = rc.rank_k_approx(A, k)
    Bk = rc.rank_k_approx(B, k)

    M_norm = norm(M - Mk)
    A_norm = norm(A - Ak)
    B_norm = norm(B - Bk)

    min_S = []
    min_obj = np.inf
    for S in list(combinations(range(n), k)):
        CM = M[:, S]
        obj_M = rc.normalized_reconstruction_error(M, CM, M_norm)
        #print(S, obj_M)
        if obj_M < min_obj:
            min_obj = obj_M
            min_S = S
        #end if

        # Delete CM to save memory
        del CM
    #end for

    CA = A[:, min_S]
    CB = B[:, min_S]

    obj_A = rc.normalized_reconstruction_error(A, CA, A_norm)
    obj_B = rc.normalized_reconstruction_error(B, CB, B_norm)

    # Delete variables to save memory
    del Mk, Ak, Bk, CA, CB

    return min_obj, obj_A, obj_B, min_S
#end brute_force()

def vanilla_obj(datasets, kk):
    print("###########################################################################")
    print("# Vanilla objective")
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
        print("# k  c    opt_M   obj_A   obj_B  obj_A-obj_B totaltime")
        for k in kk:
            temptime = time()
            min_obj, obj_A, obj_B, S = vanilla_css(M, A, B, k)
            total_time = time() - temptime

            print(" %2d %2d  %7.4f %7.4f %7.4f %12.4f %7.2f"%\
                  (k, len(S), min_obj, obj_A, obj_B, obj_A-obj_B, total_time))
            stdout.flush()

            # Delete variables to save memory
            del S
        #end for k
    #end for dataset
#end minmax_obj()

###############################################################################
#datasets = ['student-entrance', 'autism', 'heart-cleveland', 'german-credit',
#            'student-perf']
#datasets = ['autism']
#kk = [3, 4, 5, 6, 7, 8, 9, 10]
#vanilla_obj(datasets, kk)

def main():
    dataset = sys.argv[1]
    kk = [3, 4, 5, 6, 7, 8, 9, 10]
    vanilla_obj([dataset], kk)
#end main()

if __name__ == "__main__":
    main()
