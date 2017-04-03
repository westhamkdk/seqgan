import os

import numpy as np
import tensorflow as tf
import magenta.music as mm
import pickle
import random
import pandas as pd




class MIDI_IO():
    def __init__(self):
        self.note_info_path = 'note_mapping_dict.pkl'
        self.midi_training_path = "MLE_SeqGAN/save/midi_real.pkl"
        self.midi_test_path = "MLE_SeqGAN/target_generate/midi_test.pkl"
        self.midi_training_path_trans = "MLE_SeqGAN/save/midi_trans.pkl"
        self.midi_test_path_trans = "MLE_SeqGAN/target_generate/midi_trans.pkl"

        if not os.path.exists(self.note_info_path):
            self.load_all_midi_data()
        else:
            with open(self.note_info_path, "rb") as openfile:
                self.note_info_dict = pickle.load(openfile)
                self.note_info_dict_swap = dict((y, x) for x, y in self.note_info_dict.iteritems())


        print len(self.note_info_dict)

    def midi_file_to_seq(self,midi_file):
        seq = np.array([])
        try:
            melody = mm.midi_file_to_melody(midi_file, steps_per_quarter=4)
            if melody.steps_per_bar % 3 != 0:
                seq = np.array(melody._events)
                tf.logging.info('Extract melody events from7 {} file'.format(midi_file))
            else:
                tf.logging.warning('Melody of {} file has not target signature'.format(midi_file))
        except mm.MultipleTempoException as e:
            tf.logging.warning('Melody of {} file has multiple tempos'.format(midi_file))
        except mm.MultipleTimeSignatureException as e:
            tf.logging.warning('Melody of {} file has multiple signature'.format(midi_file))
        return seq


    def seq_to_midi_file(self,seq, output_file):
        melody = mm.Melody(events=seq.tolist())
        note_sequence = melody.to_sequence()
        mm.sequence_proto_to_midi_file(note_sequence, output_file)
        return seq


    def check_note_mapping_exist(self):
        if not os.path.exists(self.note_info_path):
            self.load_all_midi_data()



    def load_all_midi_data(self):
        midi_dir = os.path.expanduser("midi/")
        filenames = os.listdir(midi_dir)
        cnt = 0
        result = []
        uniques = []

        # 1216 is the longest
        longest = 0
        for filename in filenames:
            if filename.endswith(".mid"):
                seq = self.midi_file_to_seq(os.path.join(midi_dir, filename))

                # seq = seq.tolist()
                if len(seq) > 0:
                    cnt += 1

                    if len(seq) > longest:
                        longest = len(seq)

                    for i in seq:
                        if i not in uniques:
                            uniques.append(i)

                    result.append(seq)



        sorted_vals = sorted(uniques, key=abs)
        sorted_vals = map(int, sorted_vals)
        sorted_vals = np.asarray(sorted_vals)

        note_info = pd.DataFrame(data = sorted_vals, columns=['note'])

        self.note_info_dict = note_info['note'].to_dict()
        self.note_info_dict_swap = dict((y, x) for x, y in self.note_info_dict.iteritems())

        trans_list = self.trans_raw_songs_to_trans(result)

        windowed_trans_list = []
        length = 64
        stride = 32


        for midi in trans_list:
            if len(midi) > length:
                last_index = 0

                while last_index+length < len(midi):
                    windowed_trans_list.append(midi[last_index:last_index+length])
                    last_index += stride


        print len(windowed_trans_list)
        print("{} melodies extracted from {} mid files in {}"
              .format(cnt, len(filenames), midi_dir))


        with open(self.midi_training_path_trans, "w") as output_file:
            pickle.dump(windowed_trans_list, output_file)

        with open(self.note_info_path, "w") as openfile:
            pickle.dump(self.note_info_dict, openfile)





        # modified_result = []
        #
        # for arrs in result:
        #     modified_shape = np.zeros(longest)
        #     modified_shape[:arrs.shape[0]] = arrs
        #     modified_result.append(modified_shape)


        # sorted_vals = np.unique(np.asarray(modified_result))
        # sorted_vals = sorted_vals.tolist()
        # sorted_vals = sorted(sorted_vals, key=abs)
        # sorted_vals = map(int, sorted_vals)
        # sorted_vals = np.asarray(sorted_vals)
        #
        # note_info = pd.DataFrame(data=sorted_vals, columns=['note'])


        # print("{} melodies extracted from {} mid files in {}"
        #       .format(cnt, len(filenames), midi_dir))

        # random.shuffle(modified_result)
        # tr_test_index = (int)(cnt * 0.8)
        # training = modified_result[:tr_test_index]
        # test = modified_result[tr_test_index:]

        # self.note_info_dict = note_info['note'].to_dict()
        # self.note_info_dict_swap = dict((y, x) for x, y in self.note_info_dict.iteritems())

        # random.shuffle(modified_result)
        # tr_test_index = (int)(cnt * 0.8)
        # training = modified_result[:tr_test_index]
        # test = modified_result[tr_test_index:]
        #
        # with open(self.midi_training_path, "w") as output_file:
        #     pickle.dump(training, output_file)
        #
        # with open(self.midi_test_path, "w") as output_file:
        #     pickle.dump(test, output_file)
        #
        # with open(self.midi_training_path_trans, "w") as output_file:
        #     pickle.dump(self.trans_raw_songs_to_trans(training), output_file)

        # with open(self.midi_training_path, "w") as output_file:
        #     pickle.dump(modified_result, output_file)
        #
        # with open(self.midi_training_path_trans, "w") as output_file:
        #     pickle.dump(self.trans_raw_songs_to_trans(modified_result), output_file)





    def raw_note_to_trans(self, raw_note):

        result = []

        for entry in raw_note:
            result.append(self.note_info_dict_swap.get(entry))

        return result




    def trans_to_raw_note(self, trans_note):

        result = []

        for entry in trans_note:
            result.append(self.note_info_dict.get(entry))

        return result

    def trans_raw_songs_to_trans(self, raw_list):

        trans_list = []
        for midi in raw_list:
            trans_list.append(np.asarray(self.raw_note_to_trans(midi)))

        return trans_list

    def trans_trans_songs_to_raw(self, trans_list):

        raw_list = []
        for midi in trans_list:
            raw_list.append(np.asarray(self.trans_to_raw_note(midi)))

        return raw_list


if __name__ == "__main__":
    io = MIDI_IO()


    path1 = 'pretrain_small'
    with open(path1+'.pkl', 'rb') as files:
        res = pickle.load(files)
        print res

    raws =  io.trans_trans_songs_to_raw(res)


    index = 0
    for raw in raws:
        path = 'outputs/'+path1+"/"+'{}.mid'.format(index)
        io.seq_to_midi_file(raw, path)
        index+=1

    # test_raw = [-2,-2,-1,37,38,38,39,39,40,41]
    #
    # print io.raw_note_to_trans(test_raw)





