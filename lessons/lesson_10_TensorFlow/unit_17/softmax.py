#!/usr/bin/env python3

import numpy as np

def _softmax(ndarray_logits):
    nda_logits = ndarray_logits
    shape = nda_logits.shape
    dims = len(shape)

    assert(dims == 1 or dims ==2)
    assert(type(nda_logits) is np.ndarray)
    exp_logits = np.exp(nda_logits)
    exp_logits_col_sum  = np.sum(exp_logits,0) #column sum
    #FIXME: multiply every elt in exp_logits by sum of logits in the
    #            column. IWBNI you could vectorize that
    print("nda_logits = " + str(nda_logits))
    print("exp_logits = " + str(exp_logits))
    
def softmax(logits):
    assert(type(logits) is list or type(logits) is np.ndarray)
    
    if type(logits) is list:
        npa = np.array(logits)
        npa_col_vec = npa.reshape(-1, 1)
        import pdb
        return _softmax (npa_col_vec)
    else:
            return _softmax(logits)

softmax([1.0, 2.0, 3.0])
softmax(np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]]))
