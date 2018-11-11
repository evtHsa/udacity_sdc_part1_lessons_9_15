#!/usr/bin/env python3

# Solution is available in the other "solution.ipynb" 
import tensorflow as tf
import pdb

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # Calculate the softmax of the logits
    softmax = tf.nn.softmax(logit_data)
    
    with tf.Session() as sess:
        # Feed in the logit data
        output = sess.run(softmax, feed_dict={logits: logit_data})
        # softmax is the thing fetched and returned as result
        # the feed_dict overrides the logits tf_placeholder

    return output

# grader was causing grief
try:
    output = run()
    print("output = " + str(output))
except Exception as err:
    print(str(err))
