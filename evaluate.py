'''
Created on Apr 15, 2016

@author: hexiangnan
'''
import math
import operator

#todo: multi-thread evaluation
def evaluate_model(model, testRatings, K):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    hits = []
    ndcgs = []
    
    num = 0
    for rating in testRatings:
        print("#tested_ratings: %d" %num) if num % 10
        u = rating[0]
        gtItem = rating[1]
        map_item_score = {}
        # Get the score of the test item first
        maxScore = model.predict(u, gtItem)
        
        # Early stopping if there are K items larger than maxScore.
        countLarger = 0
        for i in xrange(model.num_item):
            score = model.predict(u, i)
            map_item_score[i] = score
            
            if (score>maxScore).eval() == 1:
                countLarger += 1
            if countLarger > K:
                hits.append(0)
                ndcgs.append(0)
                break
        # Generate topK rank list
        if len(map_item_score) == model.num_item:
            sorted_map = sorted(map_item_score.items, key = operator.itemgetter(1), reverse=True)
            ranklist = [sorted_map[i][0] for i in xrange(K)]
            hits.append(getHitRatio(ranklist, gtItem))
            ndcgs.append(getNDCG(ranklist, gtItem))
            
    return (hits, ndcgs)

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