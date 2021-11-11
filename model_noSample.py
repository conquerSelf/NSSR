#coding:utf-8
from __future__ import division
from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen)) #shape:(batch_size,maxlen)
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen)) #shape:(batch_size,maxlen)
        pos = self.pos

        ## ******有待改进，因为每个负样本对模型贡献程度是不一样的，如困难负样本的情况*****  比如可计算负样本评分，依据评分大小给予不同损失值权重 
        self.weight1 = 0.001 #超参数:每个负样本样例的权重大小（平均思想），与数据集稀疏度成正相关。
        
        # self.H_i = tf.Variable(tf.constant(0.01, shape=[args.hidden_units, 1]), name="hi")
        self.H_i = tf.Variable(
            tf.random_uniform([args.hidden_units, 1],
                              minval=-tf.sqrt(3.0 / args.hidden_units), maxval=tf.sqrt(3.0 / args.hidden_units),
                              name='h_i', dtype=tf.float32))
        # self.weight_u = tf.Variable(tf.constant(0.01, shape=[1,args.maxlen]), name="weight_u")

        ## 对padding值0进行mask
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)  #shape:[batch_size,maxlen,1]

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 ) #shape:batch_size*maxlen*hidden_units

            # Positional Encoding  使用的是item idx=[0,1,2,3,...,maxlen]，未使用停滞时间信息
            # pos_emb_table=[maxlen,hidden_units]
            """
            t, pos_emb_table = embedding(
                # shape:[batch_size,max_len]
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                # tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1],0,-1), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t
            """

            ##加入最后一个输入商品的嵌入表征   不该是静态的而应该是动态变化的
            # lastinput = self.seq[:,-1,:] #shape:[batch_size,hidden_units]

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq, self.weights_view = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")
                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq) #shape:batch_size x maxlen x hidden_units

        #正样本表征
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen]) #shape:(batch_size*maxlen,) 一维向量,变换维度是为了后面与seq_emb进行哈达玛乘积
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos) #shape:(batch_size*maxlen,hidden_units)
        
        ######################序列建模代表当前用户表征#############
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units]) #shape:(batch_size*maxlen,hidden_units)
        
        # print('seq_emb shape:',seq_emb.shape)
        # print('shape of item_emb_table:',item_emb_table.shape)  #shape:[itemnum + 1,hidden_size]

        # ## 1、test stage:101商品
        # self.test_item = tf.placeholder(tf.int32, shape=(101))
        # test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item) #shape:(101,hidden_units)
        # test_seq_emb = tf.reshape(seq_emb,[tf.shape(self.input_seq)[0], args.maxlen, args.hidden_units])  #shape:(batch_size,maxlen,hidden_units)
        # print('before... shape of test_seq_emb:',test_seq_emb)
        # test_seq_emb = test_seq_emb[:,-1,:]  #shape:[batch_size,hidden_size]  ### 取最后时刻的隐表征进行测试
        # test_seq_emb = tf.expand_dims(test_seq_emb,axis=1)  #shape:[batch_size,1,hidden_size]
        # test_seq_emb = tf.tile(test_seq_emb,[1,101,1])  #shape:[batch_size,101,hidden_size]
        # print('after... shape of test_seq_emb:',test_seq_emb)

        # test_item_emb = tf.expand_dims(test_item_emb,axis=0)  #shape:[1,101,hidden_units]
        # test_item_emb = tf.tile(test_item_emb,[tf.shape(self.input_seq)[0],1,1])  #shape:[batch_size,101,hidden_size]

        # dot = test_item_emb * test_seq_emb   #shape:[batch_size,101,hidden_size]
        # pre = tf.einsum('ajk,kl->ajl', dot, self.H_i)  #shape:[batch_size,101,1]
        # print('shape of pre:',pre)
        # self.test_logits = tf.squeeze(pre,squeeze_dims=2)  #shape:[batch_size,101]
        # print('shape of self.test_logits:',self.test_logits)

        ## 2、test stage:所有商品
        test_seq_emb = tf.reshape(seq_emb,[tf.shape(self.input_seq)[0], args.maxlen, args.hidden_units])  #shape:(batch_size,maxlen,hidden_units)
        print('before... shape of test_seq_emb:',test_seq_emb)
        test_seq_emb = test_seq_emb[:,-1,:]  #shape:[batch_size,hidden_size]
        test_seq_emb = tf.expand_dims(test_seq_emb,axis=1)  #shape:[batch_size,1,hidden_size]
        test_seq_emb = tf.tile(test_seq_emb,[1,itemnum+1,1])  #shape:[batch_size,itemnum + 1,hidden_size]
        print('after... shape of test_seq_emb:',test_seq_emb)

        test_item_emb = item_emb_table  #shape:[itemnum + 1,hidden_size]
        print('shape of test_item_emb:',test_item_emb.shape)

        test_item_emb = tf.expand_dims(test_item_emb,axis=0)  #shape:[1,itemnum + 1,hidden_units]
        test_item_emb = tf.tile(test_item_emb,[tf.shape(self.input_seq)[0],1,1])  #shape:[batch_size,itemnum + 1,hidden_size]
        dot = test_item_emb * test_seq_emb   #shape:[batch_size,itemnum + 1,hidden_size]
        pre = tf.einsum('ajk,kl->ajl', dot, self.H_i)  #shape:[batch_size,itemnum + 1,1]
        print('shape of pre:',pre)
        self.test_logits = tf.squeeze(pre,squeeze_dims=2)  #shape:[batch_size,itemnum + 1]
        print('shape of self.test_logits:',self.test_logits)

        # prediction layer  只对正样本进行计算评分
        self.pos_r = tf.reshape(pos_emb * seq_emb,[tf.shape(self.input_seq)[0], args.maxlen, args.hidden_units])  #shape:(batch_size,maxlen,hidden_units) 
        self.pos_r = tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i)  #将激活函数换成矩阵权重参数  shape:[batch_size,maxlen,1]
        self.pos_r = tf.reshape(self.pos_r, [-1, args.maxlen])  #shape:[batch_size,maxlen]

        #######################重点:损失函数改进#############################
        tmp = tf.reduce_sum(
            tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc', item_emb_table, item_emb_table), 0)
                          * tf.reduce_sum(tf.einsum('ab,ac->abc', seq_emb, seq_emb), 0)
                          * tf.matmul(self.H_i, self.H_i, transpose_b=True), 0), 0)
        # print('***** tmp ******:',tf.Session().run(tmp))
        self.loss = self.weight1 * tmp
        
        print('before add weight1 loss:',self.loss)
        self.loss += tf.reduce_sum((1.0 - self.weight1) * tf.square(self.pos_r) - 2.0 * self.pos_r)  # 交叉熵损失

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # reg_losses = tf.nn.l2_loss(item_emb_table)
        self.loss += sum(reg_losses)  # 交叉熵损失 + 参数正则化损失

        tf.summary.scalar('loss', self.loss)

        if reuse is None:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.is_training: False})
