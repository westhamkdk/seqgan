import sugartensor as tf
import numpy as np


class LSTM_graph(object):
    def __init__(self, x,y, num_batch, vocab_size, emb_dim, hidden_dim, max_ep = 240, infer_shape = (1,1), mode = "train"):

        self.num_batch = num_batch
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len_infer = 512
        self.max_ep = max_ep

        if mode == "train":
            self.x = x
            self.y = y

        elif mode == "infer":
            self.x = tf.placeholder(tf.int32, shape=infer_shape)
            self.y = tf.placeholder(tf.int32, shape=infer_shape)

        self.emb_x = tf.sg_emb(name='emb_x', voca_size=self.vocab_size, dim=self.emb_dim)  # (68,16)
        self.emb_y = tf.sg_emb(name='emb_y', voca_size=self.vocab_size, dim=self.emb_dim)  # (68,16)
        self.X = self.x.sg_lookup(emb=self.emb_x)  # (8,63,16)
        self.Y = self.y.sg_lookup(emb=self.emb_y)  # (8,63,16)


        if mode == "train":
            self.lstm_layer = self.X.sg_lstm(in_dim=self.emb_dim, dim=self.vocab_size)  # (8, 63, 68)

        elif mode == "infer":
            self.lstm_layer = self.X.sg_lstm(in_dim=self.emb_dim, dim=self.vocab_size, last_only=True)
            self.log_prob = tf.log(self.lstm_layer.sg_softmax())

            # next_token: select by distribution probability, preds: select by argmax
            self.next_token = tf.cast(tf.reshape(tf.multinomial(self.log_prob, 1), [1,infer_shape[0]]), tf.int32)
            self.preds = self.lstm_layer.sg_argmax()

        if mode == "train":
            self.loss = self.lstm_layer.sg_ce(target=self.y)
            self.istarget = tf.not_equal(self.y, 0).sg_float()

            self.reduced_loss = (self.loss.sg_sum()) / (self.istarget.sg_sum() + 0.0000001)
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")




    def start_training(self):
        tf.sg_train(optim='Adam', lr=0.0001, loss=self.reduced_loss,eval_metric=[self.reduced_loss], ep_size=self.num_batch, save_dir='save/train/small', max_ep=self.max_ep,
                    early_stop=False)

    def generate(self, prev_midi):
        with tf.Session() as sess:
            tf.sg_init(sess)

            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('save/train/small'))

            # KDK: choose self.next_token or self.preds
            out = sess.run(self.next_token, {self.x: prev_midi})
            return out





