import model
import numpy as np
import sugartensor as tf
import random
import time
from data_loader import Data_loader
from text_classifier import TextCNN
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle
import os
from LSTM_graph import  LSTM_graph


# Sequence length
# Variable length of sequcne should be supported



class Seq_gan():

    def __init__(self):
        #########################################################################################
        #  Generator  Hyper-parameters
        #########################################################################################
        self.PRE_EMB_DIM = 32
        self.PRE_HIDDEN_DIM = 32
        self.SEQ_LENGTH = 64
        self.PRE_START_TOKEN = 0

        self.PRE_EPOCH_NUM = 240
        # PRE_EPOCH_NUM = 5
        self.PRE_TRAIN_ITER = 1  # generator
        self.PRE_SEED = 88
        self.batch_size = 32
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
        self.negative_file = 'target_generate/midi_trans_neg.pkl'
        # eval_file = 'target_generate/midi_trans_eval.pkl'
        self.logpath = 'log/seqgan_experiment-log1.txt'

        self.generated_num = 100

        self.melody_size = 68

        self.data_loader = Data_loader()

        self.positive_x = self.data_loader.load_data(self.positive_file, self.batch_size)


    def main(self):
        random.seed(self.PRE_SEED)
        np.random.seed(self.PRE_SEED)

        assert self.PRE_START_TOKEN == 0
        # load all data

        best_score = 1000 # might be replaced as iteration continues
        lstm = LSTM_graph(self.positive_x, self.batch_size, self.melody_size, self.PRE_EMB_DIM, self.PRE_HIDDEN_DIM)
        lstm.start_training()










    if __name__ == '__main__':
        from seq_gan import Seq_gan
        seqgan = Seq_gan()
        seqgan.main()

