'''
Created on Apr 15, 2016
Implementing the Matrix Factorization with BPR ranking loss with Theano.
Steffen Rendle, et al. BPR: Bayesian personalized ranking from implicit feedback. UAI'09

@author: hexiangnan
'''
import numpy as np
import theano
import theano.tensor as T
from sets import Set
from evaluate import evaluate_model
import time

class MFbpr(object):
    '''
    BPR learning for MF model
    '''

    def __init__(self, train, test, num_user, num_item, 
                 factors, learning_rate, reg, init_mean, init_stdev):
        '''
        Constructor
        '''
        self.train = train
        self.test = test
        self.num_user = num_user
        self.num_item = num_item
        self.factors = factors
        self.learning_rate = learning_rate
        self.reg = theano.shared(value = reg, name = 'reg')
        
        # user & item latent vectors
        U_init = np.random.normal(loc=init_mean, scale=init_stdev, size=(num_user, factors))
        V_init = np.random.normal(loc=init_mean, scale=init_stdev, size=(num_item, factors))
        self.U = theano.shared(value = U_init.astype(theano.config.floatX), 
                               name = 'U', borrow = True)
        self.V = theano.shared(value = V_init.astype(theano.config.floatX),
                               name = 'V', borrow = True)
        
        # Each element is the set of items for a user, used for negative sampling
        self.items_of_user = []
        self.num_rating = 0     # number of ratings
        for u in xrange(len(train)):
            self.items_of_user.append(Set([]))
            for i in xrange(len(train[u])):
                item = train[u][i][0]
                self.items_of_user[u].add(item)
                self.num_rating += 1
        
        # variables for computing gradients
        u = T.iscalar('u')
        i_pos = T.iscalar('i_pos')
        i_neg = T.iscalar('i_neg')
        vec_u = T.vector('u')
        vec_i_pos = T.vector('vec_i_pos')
        vec_i_neg = T.vector('vec_i_neg')
        vec_u = self.U[u]
        vec_i_pos = self.V[i_pos]
        vec_i_neg = self.V[i_neg]
        lr = T.scalar('lr')
        
        # loss of the sample
        y_pos = T.dot(vec_u, vec_i_pos)
        y_neg = T.dot(vec_u, vec_i_neg)
        regularizer = self.reg * (T.dot(vec_u, vec_u) +
                                  T.dot(vec_i_pos, vec_i_pos) +
                                  T.dot(vec_i_neg, vec_i_neg))
        loss = regularizer - T.log(T.nnet.sigmoid(y_pos - y_neg))
        # gradients
        dU = T.grad(loss, vec_u)
        dV_pos = T.grad(loss, vec_i_pos)
        dV_neg = T.grad(loss, vec_i_neg)
        # SGD step: nested inc_subtensor to support update two subtensors
        sgd_update = [(self.U, T.inc_subtensor(self.U[u], -lr * dU)),
                      (self.V, T.inc_subtensor(T.inc_subtensor(self.V[i_pos], -lr * dV_pos)[i_neg], -lr * dV_neg))]
        self.sgd_step = theano.function([u, i_pos, i_neg, lr], [],
                                        updates = sgd_update)
        
    def build_model(self, maxIter, num_thread):
        # Training process
        print("Training model now.")
        for iteration in xrange(maxIter):    
            # Each training epoch
            t1 = time.time()
            for s in xrange(self.num_rating):
                # sample a user
                user = np.random.randint(0, self.num_user)
                # sample a positive item
                idx = np.random.randint(0, len(self.train[user]))
                item_pos = self.train[user][idx][0]
                # uniformly sample a negative item
                item_neg = np.random.randint(0, self.num_item)
                while item_neg in self.items_of_user[user]:
                    item_neg = np.random.randint(0, self.num_item)
                # perform a SGD step
                self.sgd_step(user, item_pos, item_neg, self.learning_rate)
            
            # check performance
            t2 = time.time()
            self.U_np = self.U.eval()
            self.V_np = self.V.eval()
            topK = 100
            (hits, ndcgs) = evaluate_model(self, self.test, topK, num_thread)
            print("Iter=%d [%.1f s] HitRatio@%d = %.3f, NDCG@%d = %.3f [%.1f s]" 
                  %(iteration, t2-t1, topK, np.array(hits).mean(), topK, np.array(ndcgs).mean(), time.time()-t2))

                            
    def predict(self, u, i):
        return np.inner(self.U_np[u], self.V_np[i])
        #return T.dot(self.U[u], self.V[i])
    
    def predict_user(self, u):
        return T.dot(self.V, self.U[u].T)
    