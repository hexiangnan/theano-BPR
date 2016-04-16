'''
Created on Apr 12, 2016

@author: hexiangnan
'''
        
def LoadRatingFile_HoldKOut(filename, splitter, K):
    """
    Each line of .rating file is: userId(starts from 0), itemId, ratingScore, time
    Each element of train is the [[item1, time1], [item2, time2] of the user, sorted by time
    Each element of test is the [user, item, time] interaction, sorted by time
    """
    train = []  
    test = []
    
    # load ratings into train.
    num_ratings = 0
    num_item = 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(splitter)
            if (len(arr) < 4):
                continue
            user, item, time = int(arr[0]), int(arr[1]), long(arr[3]) 
            if (len(train) <= user):
                train.append([])
            train[user].append([item, time])
            num_ratings += 1
            num_item = max(item, num_item)
            line = f.readline()
    num_user = len(train)
    num_item = num_item + 1
    
    # sort ratings of each user by time
    def getTime(item):
        return item[-1];
    for u in range (len(train)):
        train[u] = sorted(train[u], key = getTime)
    
    # split into train/test
    for u in range (len(train)):
        for k in range(K):
            if (len(train[u]) == 0):
                break
            test.append([u, train[u][-1][0], train[u][-1][1]])
            del train[u][-1]    # delete the last element from train
            
    # sort the test ratings by time
    test = sorted(test, key = getTime)
    
    return train, test, num_user, num_item, num_ratings
