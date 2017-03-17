import model
import numpy as np
import tensorflow as tf
import random
from gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from target_lstm import TARGET_LSTM
import pickle
import itertools

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 32
HIDDEN_DIM = 32
SEQ_LENGTH = 64
START_TOKEN = 0

PRE_EPOCH_NUM = 50  # change the pre-train epoch here
TRAIN_ITER = 1  # generator
SEED = 88
BATCH_SIZE = 16

##########################################################################################
positive_file = 'save/midi_trans.pkl'
# positive_file = 'save/real_data.txt'
negative_file = 'target_generate/generator_sample.txt'
eval_file = 'target_generate/midi_trans_eval.pkl'

generated_num = 100


class PoemGen(model.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return PoemGen(num_emb, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    #  Generated Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, "w") as fout:
        pickle.dump(generated_samples, fout)

    # with open(output_file, 'w') as fout:
    #     for poem in generated_samples:
    #         buffer = ' '.join([str(x) for x in poem]) + '\n'
    #         fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    print '>>>> generator train loss:', np.mean(supervised_g_losses)
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
    random.seed(SEED)
    np.random.seed(SEED)

    assert START_TOKEN == 0

    # load data (likelihood?)
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Likelihood_data_loader(BATCH_SIZE)
    vocab_size = 68


    # load generator with parameters
    generator = get_trainable_model(vocab_size)
    # target_params = cPickle.load(open('save/target_params.pkl'))
    target_params = initialize_parameters(68)

    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # generating synthetic data which constitute to  original data
    # generate_samples(sess, target_lstm, 64, 100, positive_file)


    gen_data_loader.create_batches(positive_file)

    log = open('log/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            buffer = str(epoch) + ' ' + str(test_loss) + '\n'
            log.write(buffer)

    generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
    likelihood_data_loader.create_batches(eval_file)
    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    print 'final pre-train epoch ', 'test_loss ', test_loss
    buffer = 'After supervised-training:' + ' ' + str(test_loss) + '\n'
    log.write(buffer)

    log.close()


if __name__ == '__main__':
    # initialize_parameters(68)
    main()

