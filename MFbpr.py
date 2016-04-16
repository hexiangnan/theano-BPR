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
        self.reg = reg
        
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
        
        # batch variables for computing gradients
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')
        lr = T.scalar('lr')
        
        # loss of the sample
        y_ui = T.dot(self.U[u], self.V[i].T).diagonal()   #1-d vector of diagonal values
        y_uj = T.dot(self.U[u], self.V[j].T).diagonal()
        regularizer = self.reg * ((self.U[u] ** 2).sum() +
                                  (self.V[i] ** 2).sum() +
                                  (self.V[j] ** 2).sum())
        loss = regularizer - T.sum(T.log(T.nnet.sigmoid(y_ui - y_uj)))
        # SGD step
        self.sgd_step = theano.function([u, i, j, lr], [],
                                        updates = [(self.U, self.U - lr * T.grad(loss, self.U)),
                                                   (self.V, self.V - lr * T.grad(loss, self.V))])
        
    def build_model(self, maxIter=100, num_thread=4, batch_size=32):
        # Training process
        print("Training MF-BPR with: learning_rate=%.2f, regularization=%.4f, factors=%d, #epoch=%d, batch_size=%d."
              %(self.learning_rate, self.reg, self.factors, maxIter, batch_size))
        for iteration in xrange(maxIter):    
            # Each training epoch
            t1 = time.time()
            for s in xrange(self.num_rating / batch_size):
                # sample a batch of users, positive samples and negative samples 
                (users, items_pos, items_neg) = self.get_batch(batch_size)
                # perform a batched SGD step
                self.sgd_step(users, items_pos, items_neg, self.learning_rate)
            
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
    
    def get_batch(self, batch_size):
        users, pos_items, neg_items = [], [], []
        for i in xrange(batch_size):
            # sample a user
            u = np.random.randint(0, self.num_user)
            # sample a positive item
            i = self.train[u][np.random.randint(0, len(self.train[u]))][0]
            # sample a negative item
            j = np.random.randint(0, self.num_item)
            while j in self.items_of_user[u]:
                j = np.random.randint(0, self.num_item)
            users.append(u)
            pos_items.append(i)
            neg_items.append(j)
        return (users, pos_items, neg_items)