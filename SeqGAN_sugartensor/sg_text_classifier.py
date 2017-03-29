import sugartensor as tf

class SG_CNN(object):



    def add_conv_pool_layer(self, x, filter_size, num_filter):
        return x.sg_conv(dim=num_filter, size = filter_size).sg_pool(size=[self.sequence_length-num_filter,1])


    def __init__(self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, input_x, input_y):

        self.sequence_length = sequence_length


        # input will be added here
        with tf.sg_context(act = 'relu'):

            for filter_size, num_filter in zip(filter_sizes, num_filters):
                x = self.add_conv_pool_layer(x, filter_size, num_filter)

            # TODO
            # Add highway
            # Add dropout
            x = x.sg_flatten().sg_dense(dim=num_classes, act='softmax', bn=False)

            loss = x.sg_ce(target = y)
            self.loss = tf.reduce_mean(loss)

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def train_