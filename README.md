# theano-BPR
Using Theano to implement the matrix factorization with BPR ranking loss, as described in:

Steffen Rendle, et al. BPR: Bayesian personalized ranking from implicit feedback. UAI'09


Required softwares: numpy, theano

Evaluation: TopK recommendation (hold-1-out evaluation), with measures Hit Ratio and NDCG

Optimizer: batched SGD

To run an example: 

  python run_example.py


  
Welcome any comments for improving the efficency! 

Author: Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

Contact: xiangnanhe@gmail.com
