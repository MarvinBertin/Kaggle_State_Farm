import tensorflow as tf
import argparse
import numpy as np
import os
import sys
import pandas as pd
sys.path.append("/home/ubuntu/tensorpack")

from tensorpack.train import TrainConfig, QueueInputTrainer
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.utils import *
from tensorpack.tfutils import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import *
from tensorpack.predict import PredictConfig, get_predict_func, DatasetPredictor

BATCH_SIZE = 32

class Model(ModelDesc):
    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 48,64, 3], 'input'),
                InputVar(tf.int32, [None], 'label')
               ]

    def _get_cost(self, input_vars, is_training):
        image, label = input_vars

        def conv(name, l, channel, stride):
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[3]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name) as scope:
                if not first:
                    b1 = BatchNorm('bn1', l, is_training)
                    b1 = tf.nn.relu(b1)
                else:
                    b1 = l
                c1 = conv('conv1', b1, out_channel, stride1)
                b2 = BatchNorm('bn2', c1, is_training)
                b2 = tf.nn.relu(b2)
                c2 = conv('conv2', b2, out_channel, 1)

                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0,0], [0,0], [0,0], [in_channel//2, in_channel//2]])

                l = c2 + l
                return l

        l = conv('conv0', image, 16, 1)
        l = BatchNorm('bn0', l, is_training)
        l = tf.nn.relu(l)
        l = residual('res1.0', l, first=True)
        for k in range(1, self.n):
            l = residual('res1.{}'.format(k), l)
        # 32,c=16

        l = residual('res2.0', l, increase_dim=True)
        for k in range(1, self.n):
            l = residual('res2.{}'.format(k), l)
        # 16,c=32

        l = residual('res3.0', l, increase_dim=True)
        for k in range(1, self.n):
            l = residual('res3.' + str(k), l)
        l = BatchNorm('bnlast', l, is_training)
        l = tf.nn.relu(l)
        # 8,c=64
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost)

        wrong = prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        tf.add_to_collection(
            MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.mul(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

class config_model(object):
    
    def __init__(self, train_or_test,random_state=1):
        self.random_state = random_state
        self.ds = dataset.KaggleStrat2(train_or_test,self.random_state)
        

    def get_data(self,train_or_test):
        isTrain = train_or_test == 'train'
        ds = dataset.KaggleStrat2(train_or_test,self.random_state)
        ds = BatchData(ds, 128, remainder=not isTrain)

        if isTrain:
            ds = PrefetchData(ds, 3, 2)
        return ds

    def get_config(self):
        # prepare dataset
        dataset_train = self.get_data('train')
        step_per_epoch = dataset_train.size()
        dataset_test = self.get_data('test')

        sess_config = get_default_sess_config(0.9)

        lr = tf.Variable(0.01, trainable=False, name='learning_rate')
        tf.scalar_summary('learning_rate', lr)

        return TrainConfig(
            dataset=dataset_train,
            optimizer=tf.train.MomentumOptimizer(lr, 0.9),
            callbacks=Callbacks([
                StatPrinter(),
                ModelSaver(),
                InferenceRunner(dataset_test,
                    [ScalarStats('cost'), ClassificationError()]),
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.03), (15, 0.01), (30, 0.001), (100, 0.0003)])
            ]),
            session_config=sess_config,
            model=Model(n=5),
            step_per_epoch=step_per_epoch,
            max_epoch=5,
        )
    
def write_submission(predictions, ids, dest):
    df = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', pd.Series(ids, index=df.index))
    df.to_csv(dest, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--iterations', help='number of iterations')
    
    args = parser.parse_args()

    basename = os.path.basename(__file__)
    

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    iterations = int(args.iterations)
    
    results = []
    
    for iteration in range(iterations):
        cm = config_model("train",random_state=iteration+1)
        logger.set_logger_dir(
            os.path.join('train2_log{iteration}'.format(iteration=iteration), basename[:basename.rfind('.')]))

        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                config = cm.get_config()
            if args.load:
                config.session_init = SaverRestore(args.load)
            if args.gpu:
                config.nr_tower = len(args.gpu.split(','))
            QueueInputTrainer(config).train()
                
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                pred_config = PredictConfig(
                model=Model(n=5),
                input_data_mapping=[0],
                session_init=SaverRestore("/home/ubuntu/tensorpack/examples/ResNet/train2_log{iteration}/Kaggle-Resnet-Strat/model-{mod}".format(iteration=iteration,mod=str(680))),
                output_var_names=['output:0']   # output:0 is the probability distribution
            )
            if args.gpu:
                config.nr_tower = len(args.gpu.split(','))

            predict_func = get_predict_func(pred_config)
            
            preds = []

            for i in xrange(0, len(cm.ds.X_test), 1000):
                x = cm.ds.X_test[i:i+1000]
                preds.append(predict_func([x])[0]
                         )

            preds = np.vstack(preds)
            results.append(preds)

    results = np.mean(results, axis=0 )
    write_submission(results, cm.ds.X_test_ids, "sub2.csv")