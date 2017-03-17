import model
import numpy as np
import sugartensor as tf
import random
import time
from gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from dis_dataloader import Dis_dataloader
from text_classifier import TextCNN
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle
import os



# Sequence length
# Variable length of sequcne should be supported

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
PRE_EMB_DIM = 32
PRE_HIDDEN_DIM = 32
SEQ_LENGTH = 64
PRE_START_TOKEN = 0

PRE_EPOCH_NUM = 240
# PRE_EPOCH_NUM = 5
PRE_TRAIN_ITER = 1  # generator
PRE_SEED = 88
PRE_BATCH_SIZE = 32
##########################################################################################

TOTAL_BATCH = 300
# TOTAL_BATCH = 800

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2

# Training parameters
dis_batch_size = 64
dis_num_epochs = 3
# dis_num_epochs = 1

# dis_alter_epoch = 50
dis_alter_epoch = 25

# kdk change here
positive_file = 'save/midi_trans.pkl'
negative_file = 'target_generate/midi_trans_neg.pkl'
# eval_file = 'target_generate/midi_trans_eval.pkl'
logpath = 'log/seqgan_experiment-log1.txt'

generated_num = 100


##############################################################################################


class PoemGen(model.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return PoemGen(num_emb, PRE_BATCH_SIZE, PRE_EMB_DIM, PRE_HIDDEN_DIM, SEQ_LENGTH, PRE_START_TOKEN)



def generate_samples(sess, trainable_model, batch_size, generated_num, output_file, epoch = -1):
    #  Generated Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, "w") as fout:
        pickle.dump(generated_samples, fout)


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def significance_test(sess, target_lstm, data_loader, output_file):
    loss = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.out_loss, {target_lstm.x: batch})
        loss.extend(list(g_loss))
    with open(output_file, 'w')as fout:
        for item in loss:
            buffer = str(item) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)



def initialize_parameters(inout_dim):

    result_list = []
    val = 32
    layers = [[inout_dim, val],

              [val, val],
              [val, val],
              [1, val],

              [val, val],
              [val, val],
              [1, val],

              [val, val],
              [val, val],
              [1, val],

              [val, val],
              [val, val],
              [1, val],

              [val, inout_dim],

              [1, inout_dim]]

    for arr_dim, layer_num in layers:
        if arr_dim > 1:
            tmp = np.random.random((arr_dim,layer_num)).astype(np.float32)
        else:
            tmp = np.random.random(layer_num,).astype(np.float32)


        result_list.append(tmp)

    result = np.array(result_list)

    return result



def main():
    random.seed(PRE_SEED)
    np.random.seed(PRE_SEED)

    assert PRE_START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(PRE_BATCH_SIZE)
    likelihood_data_loader = Likelihood_data_loader(PRE_BATCH_SIZE)
    vocab_size = 68
    dis_data_loader = Dis_dataloader()

    best_score = 1000
    # load generator with parameters
    generator = get_trainable_model(vocab_size)
    target_params = initialize_parameters(vocab_size)

    target_lstm = TARGET_LSTM(vocab_size, PRE_BATCH_SIZE, PRE_EMB_DIM, PRE_HIDDEN_DIM, SEQ_LENGTH, PRE_START_TOKEN, target_params)


    # CNNs
    with tf.variable_scope('discriminator'):
        cnn = TextCNN(
            sequence_length=SEQ_LENGTH,
            num_classes=2,
            vocab_size=vocab_size,
            embedding_size=dis_embedding_dim,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            l2_reg_lambda=dis_l2_reg_lambda)

    cnn_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
    # Define Discriminator Training procedure
    dis_global_step = tf.Variable(0, name="global_step", trainable=False)
    dis_optimizer = tf.train.AdamOptimizer(1e-4)
    dis_grads_and_vars = dis_optimizer.compute_gradients(cnn.loss, cnn_params, aggregation_method=2)
    dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # generate_samples(sess, target_lstm, 64, 10000, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open(logpath, 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            file_name = 'target_generate/pretrain_epoch'+str(epoch)+'.pkl'
            generate_samples(sess, generator, PRE_BATCH_SIZE, generated_num, file_name)
            likelihood_data_loader.create_batches(file_name)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            buffer = str(epoch) + ' ' + str(test_loss) + '\n'
            log.write(buffer)

            if epoch % 100 !=0:
                os.remove(file_name)

    file_name = 'target_generate/pretrain_finished.pkl'
    generate_samples(sess, generator, PRE_BATCH_SIZE, generated_num, file_name)
    likelihood_data_loader.create_batches(file_name)
    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    buffer = 'After pre-training:' + ' ' + str(test_loss) + '\n'
    log.write(buffer)

    file_name = 'target_generate/supervise.pkl'
    generate_samples(sess, generator, PRE_BATCH_SIZE, generated_num, file_name)
    likelihood_data_loader.create_batches(file_name)
    significance_test(sess, target_lstm, likelihood_data_loader, 'significance/supervise.txt')

    os.remove(file_name)

    print 'Start training discriminator...'
    for i in range(dis_alter_epoch):
        print 'dis_alter_epoch : ' + str(i)
        generate_samples(sess, generator, PRE_BATCH_SIZE, generated_num, negative_file)

        #  train discriminator
        dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
        dis_batches = dis_data_loader.batch_iter(
            zip(dis_x_train, dis_y_train), dis_batch_size, dis_num_epochs
        )

        for batch in dis_batches:
            try:
                x_batch, y_batch = zip(*batch)
                feed = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, step = sess.run([dis_train_op, dis_global_step], feed)
            except ValueError:
                pass

    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Reinforcement Training Generator...'
    log.write('Reinforcement Training...\n')

    for total_batch in range(TOTAL_BATCH):
        for it in range(PRE_TRAIN_ITER):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, cnn)
            feed = {generator.x: samples, generator.rewards: rewards}
            _, g_loss = sess.run([generator.g_updates, generator.g_loss], feed_dict=feed)

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:

            file_name = 'target_generate/reinforce_batch'+str(total_batch)+'.pkl'

            generate_samples(sess, generator, PRE_BATCH_SIZE, generated_num, file_name)
            likelihood_data_loader.create_batches(file_name)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = str(total_batch) + ' ' + str(test_loss) + '\n'
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            log.write(buffer)

            if total_batch % 50 != 0:
                os.remove(file_name)

            if test_loss < best_score:
                best_score = test_loss
                print 'best score: ', test_loss
                significance_test(sess, target_lstm, likelihood_data_loader, 'significance/seqgan.txt')

        rollout.update_params()

        # generate for discriminator
        print 'Start training discriminator'
        for _ in range(5):
        # for _ in range(2):

            generate_samples(sess, generator, PRE_BATCH_SIZE, generated_num, negative_file)

            dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
            dis_batches = dis_data_loader.batch_iter(zip(dis_x_train, dis_y_train), dis_batch_size, 3)

            for batch in dis_batches:
                try:
                    x_batch, y_batch = zip(*batch)
                    feed = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, step = sess.run([dis_train_op, dis_global_step], feed)
                except ValueError:
                    pass

    log.close()


if __name__ == '__main__':
    main()

