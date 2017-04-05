import sugartensor as tf
import numpy as np


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


@tf.sg_layer_func
def sg_highway(tensor, opt):
    input_ = tensor
    output = input_
    for idx in xrange(opt.layer_size):
        g = tf.nn.relu(linear(input_, opt.size, scope='highway_lin_%d' % idx))

        t = tf.sigmoid(linear(input_, opt.size, scope='highway_gate_%d' % idx) + opt.bias)

        output = t * g + (1. - t) * input_
        input_ = output
    return output

tf.sg_inject_func(sg_highway)


class CNN_graph(object):
    def __init__(self, x,y,sequence_length, num_classes, vocab_size, embed_size, filter_sizes, num_filters, epoch,num_batch, lg_reg_lambda=0.0 , mode="train"):

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.epoch = epoch
        self.num_batch = num_batch

        if mode == "train":
            self.x, self.y = x,y

        self.emb_x = tf.sg_emb(name='CNN_emb_x', voca_size=self.vocab_size, dim=self.embed_size)
        self.emb_y = tf.sg_emb(name='CNN_emb_y', voca_size=self.num_classes, dim=self.embed_size)
        self.X = self.x.sg_lookup(emb=self.emb_x)  # (8,63,16)
        self.Y = self.y.sg_lookup(emb=self.emb_y)  # (8,63,16)



        pooled = []
        with tf.sg_context(act='relu'):
            isFirst = True
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                # layer = self.X.sg_conv1d(size= filter_size, dim=num_filter, pad='VALID').sg_pool1d(size=2, stride=1)
                layer = self.X.sg_conv1d(size= filter_size, dim=num_filter, pad='VALID')
                layer = layer.sg_pool1d(size=sequence_length-filter_size+1, stride = 1)

                pooled.append(layer)

        self.h_pool = tf.concat(pooled, 2)
        num_filters_total = sum(num_filters)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


        # self.X = self.X.sg_flatten().sg_highway(layer_size = 1, size = self.X.get_shape()[1], bias = 0).sg_bypass(dout=10)
        print self.h_pool_flat
        print self.h_pool_flat.get_shape()[1]
        self.highway = self.h_pool_flat.sg_highway(layer_size = 1, size = self.h_pool_flat.get_shape()[1], bias=0).sg_bypass(dout=0.9)
        self.logit = self.highway.sg_dense(dim=num_classes, act='softmax')

        print self.logit
        self.predictions = self.logit.sg_argmax()
        print self.y

        if mode == "train":
            self.loss = self.logit.sg_ce(target=self.y, one_hot=True)

            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            tf.sg_summary_loss(self.accuracy, "accuracy")


    def start_training(self):
        tf.sg_train(optim='Adam', lr=1e-4, loss=self.loss,eval_metric=[self.loss], ep_size=self.num_batch, save_dir='pretrain/disc', max_ep=self.epoch,
                    early_stop=False)




