# theano-BPR
============
Using Theano to implement the matrix factorization with BPR ranking loss, as described in:

> Steffen Rendle, et al. [BPR: Bayesian personalized ranking from implicit feedback](http://arxiv.org/pdf/1205.2618.pdf) UAI'09

Optimizer: Batched SGD

Usage: 
------
    Required softwares: numpy, theano
    $ python run_example.py
    
Evaluation: 
-----------
TopK recommendation (hold-1-out evaluation), with measures Hit Ratio and NDCG.
More details about the evaluation can be found in: 

> Xiangnan He, et al. [Fast Matrix Factorization for Online Recommendation
with Implicit Feedback](http://www.comp.nus.edu.sg/~xiangnan/papers/sigir16-eals-draft.pdf) SIGIR'16

**Welcome any comments for improving the efficency!**

## Author:

Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

Contact: xiangnanhe@gmail.com
