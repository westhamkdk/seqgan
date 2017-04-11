import numpy as np
from re import compile as _Re
import pickle
import sugartensor as tf




class Data_loader():
    def load_data(self, data_file, batch_size):
        with open(data_file, "r") as output_file:
            # song sequence : 64
            # length of list : 1024
            raw_data = np.asarray(np.load(output_file))

            print raw_data.shape

            # actual sequence length : 63
            raw_x = raw_data[:,:-1]
            raw_y = raw_data[:,1:]




            x_q, y_q = tf.train.slice_input_producer([tf.convert_to_tensor(raw_x, tf.int32), tf.convert_to_tensor(raw_y, tf.int32)])




            X, Y = tf.train.shuffle_batch([x_q, y_q],
                                   batch_size = batch_size,
                                   capacity= batch_size*64,
                                   min_after_dequeue=batch_size*32)


            return X, Y

    def load_data_and_labels(self, positive_file, negative_file, batch_size1):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """


        with open(positive_file, "rb") as output_file:
            positive_examples = pickle.load(output_file)
            positive_len = len(positive_examples[0])
            print positive_len

        with open(negative_file, "rb") as output_file:
            negative_examples = np.asarray(np.load(output_file))
            negative_examples = negative_examples[:,:positive_len]

        # Split by words

        # x_text = positive_examples + negative_examples
        x_text = np.concatenate([positive_examples, negative_examples], 0)


        # Generate labels
        positive_labels = np.asarray([np.asarray([0,1]) for _ in positive_examples])
        negative_labels = np.asarray([np.asarray([1,0]) for _ in negative_examples])
        y = np.concatenate([positive_labels, negative_labels], 0)


        x_text = np.array(x_text)
        y = np.array(y)

        x_q, y_q = tf.train.slice_input_producer(
            [tf.convert_to_tensor(x_text, tf.int32), tf.convert_to_tensor(y, tf.int32)])



        X,Y = tf.train.shuffle_batch([x_q, y_q],
                                      batch_size=batch_size1,
                                      capacity=batch_size1*64,
                                      min_after_dequeue=batch_size1*32)

        return X,Y







