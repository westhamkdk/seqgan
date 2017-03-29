import sugartensor as tf


class LSTM_graph(object):
    def __init__(self, x, num_batch, vocab_size, emb_dim, hidden_dim):
        self.x = x
        self.num_batch = num_batch
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size


    def start_training(self):
        self.emb_x = tf.sg_emb(name='emb_x', voca_size = self.vocab_size, dim=self.hidden_dim) #(68,32)
        X = self.x.sg_lookup(emb = self.emb_x) # (32,64,32)
        print X
        lstm_layer = X.sg_lstm(in_dim=self.emb_dim, dim=self.hidden_dim) # (32, 64, 32)



        # logits = lstm_layer.sg_dense(in_dim = self.hidden_dim, dim=self.vocab_size)
        logits = lstm_layer.sg_dense(dim=self.vocab_size)
        print logits

        preds = logits.sg_argmax()

        loss = logits.sg_ce(target = self.x)
        istarget = tf.not_equal(self.x,0).sg_float()
        reduced_loss = (loss.sg_sum()) / (istarget.sg_sum() + 0.0000001)
        tf.sg_summary_loss(reduced_loss, "reduced_loss")

        tf.sg_train(optim='Adam', rl = 0.001, loss=reduced_loss, ep_size = self.num_batch, save_dir = 'save/train', max_ep = 30)



