import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A sample CNN for classification
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,
                 num_classes,
                 batch_size,
                 sequence_length,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 decay_rate_big=0.5,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 multi_label_flag=False,
                 clip_gradients=5.0,
                 is_training=True,
                 l2_reg_lambda=0.0):


        #定义超参数
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate*decay_rate_big)
        self.filter_sizes = filter_sizes
        self.num_features = num_filters
        self.initializer = initializer
        self.total_filters = self.num_features * len(self.filter_sizes)
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients


        #定义输入输出以及dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.dropout = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.iter = tf.placeholder(tf.int32)
        self.tst = tf.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.constant(1))
        self.b1 = tf.Variable(tf.ones([self.num_features]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_features]) / 10)
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weight()
        self.logits = self.inference()
        self.possibility = tf.nn.sigmoid(self.logits)

        if not self.is_training:
            return
        if self.multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.multi_loss(l2_reg_lambda)
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss(l2_reg_lambda)
        self.train_op = self.train()
        if not self.multi_label_flag:
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            print("self.predictions:",self.predictions)
            correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="Accuracy")

    def instantiate_weight(self):
        "定义权重"
        with tf.name_scope('embedding'):
            self.embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embedding_size])
            self.W_p = tf.get_variable("W_projection", shape=[self.total_filters, self.num_classes])
            self.b_p = tf.get_variable("b_projection", shape=[self.num_classes])

    def inference(self):
        # model的构成：embedding ==> conv ==> pooling
        self.embedding_words = tf.nn.embedding_lookup(self.embedding, self.input_x)
        self.embedding_words_expand = tf.expand_dims(self.embedding_words, -1)

        pooled_outpus = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv_pooling_%s"%filter_size):
                # 此时的输入：NHWC=[batch_size, sequence_len, embedding_size, 1]
                # 依据卷积核的大小创建出一个合适的filter, [filter_height, feature_width, input_channels, outpus_channels] = [filter_size, embedding_size, 1, num_features]
                # 输出的大小就变为[batch_size, (sequence_len-filter_size)/stride+1, 1, num_filters]
                filter = tf.get_variable(name="filter-%s"%filter_size,
                                         shape=[filter_size, self.embedding_size, 1, self.num_features],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(input=self.embedding_words_expand,
                                    filter=filter,
                                    strides=[1,1,1,1],
                                    padding='VALID',
                                    name='conv')
                # bn一下，并通过一个relu函数
                conv, self.update_ema = self.batchnorm(conv, self.tst, self.iter, self.b1)
                b = tf.get_variable("b-%s"%filter_size, [self.num_features])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), 'relu')

                # max_pooling
                pooled = tf.nn.max_pool(value=h,
                                        ksize=[1, self.sequence_length-filter_size+1, 1, 1],
                                        strides=[1,1,1,1],
                                        padding='VALID',
                                        name='pool')
                pooled_outpus.append(pooled)

        self.pool = tf.concat(pooled_outpus, 3)
        self.pool_flat = tf.reshape(self.pool, [-1, self.total_filters])


        ## dropout
        with tf.name_scope('dropout'):
            self.pool_drop = tf.nn.dropout(self.pool_flat, keep_prob=self.dropout)

        self.pool_drop = tf.layers.dense(self.pool_drop, self.total_filters, activation=tf.nn.tanh, use_bias=True)

        # logits 一层线性函数
        with tf.name_scope("outputs"):
            logits = tf.matmul(self.pool_drop, self.W_p) + self.b_p
        return  logits

    @staticmethod
    def batchnorm(Y_logits, is_test, iteration, offset, convolutional=False):
        """
        一个很棒的数据trick
        :param Y_logits:
        :param is_test:
        :param iteration:
        :param offset:
        :param convolution:
        :return:
        """
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Y_logits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Y_logits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        ybn = tf.nn.batch_normalization(Y_logits, m, v, offset, None, bnepsilon)

        return ybn, update_moving_averages

    def multi_loss(self, l2_lambda):
        with tf.name_scope("multi_loss"):
            # `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            # print("sigmoid_cross_entropy_with_logits loss:", losses)
            losses = tf.reduce_sum(losses)
            loss = tf.reduce_mean(losses)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss += loss+l2_loss
        return loss

    def loss(self, l2_lambda):
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        return  l2_loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer='Adam', clip_gradients=self.clip_gradients)

        return train_op




#test started. toy task: given a sequence of data. compute it's label: sum of its previous element,itself and next element greater than a threshold, it's label is 1,otherwise 0.
#e.g. given inputs:[1,0,1,1,0]; outputs:[0,1,1,1,0].
#invoke test() below to test the model in this toy task.
def test():
    #below is a function test; if you use this for text classifiction, you need to transform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=5
    learning_rate=0.001
    batch_size=8
    decay_steps=1000
    decay_rate=0.95
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=1.0 #0.5
    filter_sizes=[2,3,4]
    num_filters=128
    multi_label_flag=True
    textRNN=TextCNN(num_classes=num_classes,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    vocab_size=vocab_size,
                    embedding_size=embed_size,
                    filter_sizes=filter_sizes,
                    num_filters=num_filters,
                    learning_rate=learning_rate,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    is_training=is_training,
                    multi_label_flag=multi_label_flag)
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       for i in range(500):
           input_x=np.random.randn(batch_size,sequence_length) #[None, self.sequence_length]
           input_x[input_x>=0]=1
           input_x[input_x <0] = 0
           input_y_multilabel=get_label_y(input_x)
           loss,possibility,W_projection_value,_=sess.run([textRNN.loss_val,textRNN.possibility,textRNN.W_p,textRNN.train_op],
                                                    feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y_multilabel,textRNN.dropout:dropout_keep_prob})
           print(i,"loss:",loss,"-------------------------------------------------------")
           print("label:",input_y_multilabel);print("possibility:",possibility)

def get_label_y(input_x):
    length=input_x.shape[0]
    input_y=np.zeros((input_x.shape))
    for i in range(length):
        element=input_x[i,:] #[5,]
        result=compute_single_label(element)
        input_y[i,:]=result
    return input_y

def compute_single_label(listt):
    result=[]
    length=len(listt)
    for i,e in enumerate(listt):
        previous=listt[i-1] if i>0 else 0
        current=listt[i]
        next=listt[i+1] if i<length-1 else 0
        summ=previous+current+next
        if summ>=2:
            summ=1
        else:
            summ=0
        result.append(summ)
    return result

test()