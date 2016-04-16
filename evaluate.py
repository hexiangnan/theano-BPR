'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_K = None

def evaluate_model(model, testRatings, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _K
    _model = model
    _testRatings = testRatings
    _K = K
    num_rating = len(testRatings)
    
    pool = multiprocessing.Pool(processes=num_thread)
    res = pool.map(eval_one_rating, range(num_rating))
    pool.close()
    pool.join()
    
    hits = [r[0] for r in res]
    ndcgs = [r[1] for r in res]
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    hr = ndcg = 0
    u = rating[0]
    gtItem = rating[1]
    map_item_score = {}
    # Get the score of the test item first
    maxScore = _model.predict(u, gtItem)
    # Early stopping if there are K items larger than maxScore.
    countLarger = 0
    for i in xrange(_model.num_item):
        early_stop = False
        score = _model.predict(u, i)
        map_item_score[i] = score
        
        if score > maxScore:
            countLarger += 1
        if countLarger > _K:
            hr = ndcg = 0
            early_stop = True
            break
    # Generate topK rank list
    if early_stop == False:
        ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)
        
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
