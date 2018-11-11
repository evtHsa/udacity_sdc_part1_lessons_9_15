#!/usr/bin/env python3

import numpy as np

def _softmax(ndarray_logits):
    nda_logits = ndarray_logits
    shape = nda_logits.shape
    dims = len(shape)
    assert(dims == 1 or dims ==3)
    print("dims = " + str(len(shape)))
    
def softmax(logits):
    print("type(logits_ = " + str(type(logits)))
    print("logits = " + str(logits))
    assert(type(logits) is list or type(logits) is np.ndarray)
    
    if type(logits) is list:
        return _softmax (np.array(logits))
    else:
            return _softmax(logits)

softmax([1.0, 2.0, 3.0])
softmax(np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]]))
