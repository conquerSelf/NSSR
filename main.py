#coding:utf-8
import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
# from model import Model
from model_noSample import Model
from tqdm import tqdm

# from util import *    #测试集为101个采样物品

from util_noSample import *    #测试集为全部物品

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print 'average sequence length: %.2f' % (cc / len(user_train))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())

import shelve
similar_items = None
# similar_items = shelve.open('./data/ml_1m_50d_similar_items_100to400','r')
# item_emb_table = None

from sklearn.metrics.pairwise import cosine_similarity
def itemID_most_similar(items_emb,id,low,high):
    id_feature = items_emb[id-1]
    seimilar_list = []
    for i,otherItem_feature in enumerate(items_emb):
        if i != (id-1):
            ## 根据两个特征向量计算余弦相似度大小
            seimilar_list.append(cosine_similarity(id_feature.reshape(1,-1),otherItem_feature.reshape(1,-1))[0][0])
    k_similar_items = np.argsort(np.array(seimilar_list))[-high:-low]  #这里改为取中间段的商品会更好吗？比如 [-400:-100]
    return [x + 1 for x in k_similar_items]

T = 0.0
t0 = time.time()
sampler = WarpSampler(similar_items, user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
weights_view = None
# flag = 0
try:
    for epoch in range(1, args.num_epochs + 1):
        # print('epoch number:',epoch)
        # sampler = WarpSampler(similar_items, user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            
            ## 原论文输入
            auc, loss, _, item_emb_table = sess.run([model.auc, model.loss, model.train_op, model.item_emb_table],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})

            ## 非采样模型输入
            # loss, _ , weights_view = sess.run([model.loss, model.train_op,model.weights_view],
            #                         {model.u: u, model.input_seq: seq, model.pos: pos,
            #                          model.is_training: True})
        if epoch % 20 == 0:
            t1 = time.time() - t0
            T += t1
            print 'Evaluating',
            t_test = evaluate(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print ''
            # print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            # epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])

            print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f),valid (NDCG@50: %.4f, HR@50: %.4f), valid (NDCG@100: %.4f, HR@100: %.4f),valid (NDCG@200: %.4f, HR@200: %.4f),\
            test (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@50: %.4f, HR@50: %.4f), test (NDCG@100: %.4f, HR@100: %.4f), test (NDCG@200: %.4f, HR@200: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1],t_valid[2], t_valid[3],t_valid[4], t_valid[5],t_valid[6], t_valid[7], \
            t_test[0], t_test[1],t_test[2], t_test[3],t_test[4],t_test[5],t_test[6], t_test[7])

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
        # if epoch % 500 == 0:
        #     # flag = 1
        #     print('epoch:',epoch)
        #     for i in range(args.maxlen-20,args.maxlen):
        #         print('weights_view[0][%d][args.maxlen-20:args.maxlen]:',i)
        #         print('shape of weights_view[0][i]',len(weights_view[0][i]))
        #         print(weights_view[0][i][args.maxlen-20:args.maxlen])  #取用户最近交互的20个物品位置权重值
                # print('weights_view[0][i][:20]:',weights_view[0][i][:20])
            # save_weights_view = open('./save_weights_view.txt','w')
            # save_weights_view.write('epoch:'+str(epoch))
            # save_weights_view.write(str(weights_view[0][0][-20:]))   
            # save_weights_view.close()
        """
        sampler.close()
        ## 动态更新每个商品相似的其他商品
        if epoch % 5 == 0:
            # print('dynamic updating......')
            # print('item_emb_table shape:',item_emb_table.shape)
            similar_items = None
            items_emb = item_emb_table[1:, :]
            print('items_emb.shape[0]',items_emb.shape[0])
            if os.path.exists('./data/ml_1m_50d_similar_items_100to400'):
                os.remove('./data/ml_1m_50d_similar_items_100to400')
            similar_items = shelve.open('./data/ml_1m_50d_similar_items_100to400', flag='c', protocol=2, writeback=False)
            for id in range(1,items_emb.shape[0]+1):
                k_similar = itemID_most_similar(items_emb,id,100,400)
                tmp = []
                for j in k_similar:
                    tmp.append(j)
                similar_items[str(id)] = tmp
            # print('similar_items:',similar_items)
        """
except:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
print("Done")
