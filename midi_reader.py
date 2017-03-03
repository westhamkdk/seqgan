import fnmatch
import os
import threading

import numpy as np
import tensorflow as tf

import midi_io


def find_files(directory, pattern='*.mid'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def melody_to_represenatation(midi, encoding):
    if encoding == 'time_single':
        midi = midi.reshape(-1, 1)
        midi += 2
    elif encoding == 'time_sep':
        pitches = [max(midi[0], -1)]
        for i, e in enumerate(midi[1:]):
            pitches.append(e if e >= 0 else pitches[i])
        pitches = np.array(pitches)
        beats = np.where(midi >= 0, 0, -midi)
        midi = {'pitches': pitches.reshape(-1, 1),
                'beats': beats.reshape(-1, 1)}
    return midi


def load_generic_midi(directory, encoding):
    '''Generator that yields midi notes from the directory.'''
    files = find_files(directory)
    for filename in files:
        midi = midi_io.midi_file_to_seq(filename)
        if len(midi) < 1:
            continue
        midi = melody_to_represenatation(midi, encoding)
        yield midi, filename


class MidiReader(object):
    '''Generic background audio reader that preprocesses midi files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 midi_dir,
                 coord,
                 sample_size,
                 queue_size=256,
                 encoding='time_single'):
        self.midi_dir = midi_dir
        self.coord = coord
        self.sample_size = sample_size
        self.threads = []
        self.encoding = encoding
        if self.encoding == 'time_single':
            self.sample_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
            self.queue = tf.PaddingFIFOQueue(
                queue_size, ['int32'], shapes=[(None, 1)])
            self.enqueue = self.queue.enqueue([self.sample_placeholder])
        elif self.encoding == 'time_sep':
            self.pitches_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
            self.beats_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
            self.queue = tf.PaddingFIFOQueue(
                queue_size, ['int32', 'int32'], shapes=[(None, 1), (None, 1)])
            self.enqueue = self.queue.enqueue(
                [self.pitches_placeholder, self.beats_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(midi_dir):
            raise ValueError("No midi files found in '{}'.".format(midi_dir))

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_midi(self.midi_dir, self.encoding)
            for midi, filename in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.sample_size and self.encoding == 'time_single':
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, midi)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        buffer_ = buffer_[self.sample_size:]
                else:
                    if self.encoding == 'time_single':
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: midi})
                    elif self.encoding == 'time_sep':
                        sess.run(self.enqueue,
                                 feed_dict={self.pitches_placeholder: midi['pitches'],
                                            self.beats_placeholder: midi['beats']})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads


if __name__ == "__main__":
    midi_dir = os.path.expanduser("~/midi-wavenet/data/midi")

    coord = tf.train.Coordinator()
    reader = MidiReader(midi_dir, coord,
                        sample_size=None, queue_size=4, encoding='time_sep')
    midi_batch = reader.dequeue(2)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.initialize_all_variables()
    sess.run(init)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    try:
        encoded = sess.run(midi_batch)
        print encoded
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        coord.request_stop()
        coord.join(threads)
