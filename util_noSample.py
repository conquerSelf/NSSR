#coding:utf-8
import sys
import copy
import random
import numpy as np
from collections import defaultdict


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)  #使用字典类型存储用户物品pair
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]  #留1法进行验证和测试
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    # print('util_noSample start evaluating......')
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG_10 = 0.0
    HT_10 = 0.0

    NDCG_50 = 0.0
    HT_50 = 0.0

    NDCG_100 = 0.0
    HT_100 = 0.0

    NDCG_200 = 0.0
    HT_200 = 0.0

    valid_user = 0.0

    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        #2、对所有商品进行概率预测
        predictions = -model.predict(sess, [u], [seq])  #shape:[batch_size,itemnum+1]
        
        # print('######shape of predictions:#######',predictions.shape)
        predictions = predictions[0]  #shape:[itemnum+1]
        # print('predictions type:',type(predictions))
        # print('predictions shape:',predictions.shape)
        
        #将padding值0的预测结果删掉  
        predictions = predictions[1:]  # shape:[batch_size,itemnum]
        # print('test[u][0]:',test[u][0])

        #argsort()函数返回的是数组从小到大排序后对应的数组索引
        ans = predictions.argsort().argsort()
        rank = ans[test[u][0]-1]  #返回正样本的实际排名-1,即排名值为0,1,2,3...

        # 可选:Caser论文trick:将用户u交互过的物品集合不作为最后的指标计算
        
        rated = train[u]
        tmp_rank = rank
        for item in rated:
            if ans[item-1] < rank:
                tmp_rank = tmp_rank - 1
        rank = tmp_rank
        
        
        valid_user += 1
        
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)   #实质计算的是DCG而不是NDCG
            HT_10 += 1

        if rank < 50:
            NDCG_50 += 1 / np.log2(rank + 2)
            HT_50 += 1

        if rank < 100:
            NDCG_100 += 1 / np.log2(rank + 2)
            HT_100 += 1
        
        if rank < 200:
            NDCG_200 += 1 / np.log2(rank + 2)
            HT_200 += 1

        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user,NDCG_50 / valid_user, HT_50 / valid_user,NDCG_100 / valid_user, HT_100 / valid_user,NDCG_200 / valid_user, HT_200 / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG_10 = 0.0
    HT_10 = 0.0

    NDCG_50 = 0.0
    HT_50 = 0.0

    NDCG_100 = 0.0
    HT_100 = 0.0
    
    NDCG_200 = 0.0
    HT_200 = 0.0
    HT = 0.0
    
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        ## 所有商品
        predictions = -model.predict(sess, [u], [seq])
        predictions = predictions[0]

        #将padding值0的预测结果删掉  
        predictions = predictions[1:]  # shape:[batch_size,itemnum]
        # print('test[u][0]:',test[u][0])

        #argsort()函数返回的是数组从小到大排序后对应的数组索引
        ans = predictions.argsort().argsort()
        rank = ans[test[u][0]-1]  #返回正样本的实际排名-1,即排名值为0,1,2,3...

        # 可选:Caser论文trick:将用户u交互过的物品集合不作为最后的指标计算
        rated = train[u]
        tmp_rank = rank
        for item in rated:
            if ans[item-1] < rank:
                tmp_rank = tmp_rank - 1
        rank = tmp_rank

        valid_user += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1

        if rank < 50:
            NDCG_50 += 1 / np.log2(rank + 2)
            HT_50 += 1

        if rank < 100:
            NDCG_100 += 1 / np.log2(rank + 2)
            HT_100 += 1
        
        if rank < 200:
            NDCG_200 += 1 / np.log2(rank + 2)
            HT_200 += 1

        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user,NDCG_50 / valid_user, HT_50 / valid_user,NDCG_100 / valid_user, HT_100 / valid_user,NDCG_200 / valid_user, HT_200 / valid_user

