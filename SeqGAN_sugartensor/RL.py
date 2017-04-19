
import sugartensor as tf
from LSTM_graph import  LSTM_graph
import numpy as np
from seq_gan import Seq_gan



class RL(object):
    def __init__(self, lstm_graph, update_rate, seqgan):
        self.update_rate = update_rate
        self.lstm_graph = lstm_graph


        self.origin_lstm = self.lstm_graph.lstm_layer
        print self.origin_lstm

        self.RL_lstm = self.origin_lstm.sg_identity()
        print self.RL_lstm

        self.seqgan = seqgan


    def get_reward(self, sess, input_x, rollout_num, cnn):
        rewards = []

        for i in range(rollout_num):
            result = self.seqgan.generate_samples()













