import model
import numpy as np
import sugartensor as tf
import random
import time
from data_loader import Data_loader

from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle
import os
from LSTM_graph import  LSTM_graph
from CNN_graph import CNN_graph


# Sequence length
# Variable length of sequcne should be supported


tf.sg_verbosity(10)



class Seq_gan():

    def __init__(self):
        #########################################################################################
        #  Generator  Hyper-parameters
        #########################################################################################
        self.PRE_EMB_DIM = 32
        self.PRE_HIDDEN_DIM = 32
        self.SEQ_LENGTH = 64
        self.PRE_START_TOKEN = 0

        # self.PRE_EMB_DIM = 16
        # self.PRE_HIDDEN_DIM = 32
        # self.SEQ_LENGTH = 64

        self.PRE_EPOCH_NUM = 240
        # PRE_EPOCH_NUM = 5
        self.PRE_TRAIN_ITER = 1  # generator
        self.PRE_SEED = 88
        self.batch_size = 8
        ##########################################################################################

        self.TOTAL_BATCH = 300
        # TOTAL_BATCH = 800

        #########################################################################################
        #  Discriminator  Hyper-parameters
        #########################################################################################
        self.dis_embedding_dim = 64
        self.dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        self.dis_dropout_keep_prob = 0.75
        self.dis_l2_reg_lambda = 0.2

        # Training parameters
        self.dis_num_epochs = 3
        # dis_num_epochs = 1

        # dis_alter_epoch = 50
        self.dis_alter_epoch = 25

        # kdk change here
        self.positive_file = 'save/midi_trans.pkl'
        self.negative_file = 'target_generate/pretrain_small.pkl'
        # eval_file = 'target_generate/midi_trans_eval.pkl'
        self.logpath = 'log/seqgan_experiment-log1.txt'

        self.generated_num = 40

        self.melody_size = 68

        self.data_loader = Data_loader()

        self.positive_x, self.positive_y = self.data_loader.load_data(self.positive_file, self.batch_size)


    def main(self):
        random.seed(self.PRE_SEED)
        np.random.seed(self.PRE_SEED)

        assert self.PRE_START_TOKEN == 0

        # # pretrain generator
        # best_score = 1000 # might be replaced as iteration continues
        # lstm = LSTM_graph(self.positive_x, self.positive_y, self.batch_size, self.melody_size, self.PRE_EMB_DIM, self.PRE_HIDDEN_DIM)
        # lstm.start_training()
        #
        # # generate data
        # generated_samples = []
        # for _ in range((int(self.generated_num/self.batch_size))):
        #     first_token = 1
        #     result = [[first_token],[first_token],[first_token]]
        #     result = np.asarray(result)
        #
        #     for i in range(128):
        #         print i
        #         tf.reset_default_graph()
        #         lstm_eval = LSTM_graph(self.positive_x, self.positive_y, self.batch_size, self.melody_size,
        #                                self.PRE_EMB_DIM,
        #                                self.PRE_HIDDEN_DIM, infer_shape=result.shape, mode="infer")
        #         last_token = lstm_eval.generate(result)
        #         last_token = np.transpose(last_token)
        #
        #         result = np.column_stack((result, last_token))
        #     generated_samples.extend(result)
        #
        # print generated_samples
        # file_name = 'target_generate/pretrain_small.pkl'
        # with open(file_name, "w") as fout:
        #     pickle.dump(generated_samples, fout)
        #
        X,Y = self.data_loader.load_data_and_labels(self.positive_file, self.negative_file, self.batch_size)
        #self, x, y, sequence_length, num_classes, vocab_size, embed_size, filter_sizes, num_filters, epoch, lg_reg_lambda = 0.0, mode = "train"):
        cnn = CNN_graph(X,Y,self.SEQ_LENGTH, 2, self.melody_size, self.dis_embedding_dim, self.dis_filter_sizes, self.dis_num_filters, self.dis_num_epochs, self.batch_size)
        cnn.start_training()






    if __name__ == '__main__':
        from seq_gan import Seq_gan
        seqgan = Seq_gan()
        seqgan.main()


