import sugartensor as tf
import numpy as np


class LSTM_graph(object):
    def __init__(self, x,y, num_batch, vocab_size, emb_dim, hidden_dim, mode = "train"):

        self.num_batch = num_batch
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len_infer = 512

        if mode == "train":
            self.x = x
            self.y = y

        elif mode =="infer":
            self.x = tf.placeholder(tf.int32, shape=(1, None))
            self.y = tf.placeholder(tf.int32, shape=(1, None))



        self.emb_x = tf.sg_emb(name='emb_x', voca_size=self.vocab_size, dim=self.emb_dim)  # (68,16)
        self.X = self.x.sg_lookup(emb=self.emb_x)  # (8,63,16)
        # print self.X
        #
        self.emb_y = tf.sg_emb(name='emb_y', voca_size=self.vocab_size, dim=self.emb_dim)  # (68,16)
        self.Y = self.y.sg_lookup(emb=self.emb_y)  # (8,63,16)

        self.lstm_layer = self.X.sg_lstm(in_dim=self.emb_dim, dim=self.vocab_size)  # (8, 63, 68)
        # self.lstm_layer = self.X.sg_lstm(in_dim=self.emb_dim, dim=self.vocab_size, last_only = True)
        self.preds = self.lstm_layer.sg_argmax()

        if mode == "train":
            self.loss = self.lstm_layer.sg_ce(target=self.y)
            self.istarget = tf.not_equal(self.y, 0).sg_float()

            self.reduced_loss = (self.loss.sg_sum()) / (self.istarget.sg_sum() + 0.0000001)
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")

    def start_training(self):
        tf.sg_train(optim='Adam', lr=0.0001, loss=self.reduced_loss,eval_metric=[self.reduced_loss], ep_size=self.num_batch, save_dir='pre_small', max_ep=240,
                    early_stop=False)

    def generate(self):
        with tf.Session() as sess:
            tf.sg_init(sess)

            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('save/train'))
            print ("Restored!")


            for i in range(5):
                norm = [[2,3]]
                norm = np.asarray(norm)
                out = sess.run(self.preds, {self.x : norm,self.y :norm})
                print out





