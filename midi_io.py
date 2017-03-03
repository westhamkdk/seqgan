import os

import numpy as np
import tensorflow as tf
import magenta.music as mm


def midi_file_to_seq(midi_file):
    seq = np.array([])
    try:
        melody = mm.midi_file_to_melody(midi_file, steps_per_quarter=4)
        if melody.steps_per_bar % 3 != 0:
            seq = np.array(melody._events)
            tf.logging.info('Extract melody events from {} file'.format(midi_file))
        else:
            tf.logging.warning('Melody of {} file has not target signature'.format(midi_file))
    except mm.MultipleTempoException as e:
        tf.logging.warning('Melody of {} file has multiple tempos'.format(midi_file))
    except mm.MultipleTimeSignatureException as e:
        tf.logging.warning('Melody of {} file has multiple signature'.format(midi_file))
    return seq


def seq_to_midi_file(seq, output_file):
    melody = mm.Melody(events=seq.tolist())
    note_sequence = melody.to_sequence()
    mm.sequence_proto_to_midi_file(note_sequence, output_file)
    return seq


if __name__ == "__main__":
    # midi_file_to_seq(os.path.expanduser("~/umm/data/midi/children_song/hicdic.mid"))

    # midi_dir = os.path.expanduser("~/umm/data/midi/children_song")
    midi_dir = os.path.expanduser("~/midi-wavenet/data/midi")
    filenames = os.listdir(midi_dir)
    cnt = 0
    for filename in filenames:
        if filename.endswith(".mid"):
            seq = midi_file_to_seq(os.path.join(midi_dir, filename))
            if len(seq) > 0:
                cnt += 1
    print("{} melodies extracted from {} mid files in {}"
          .format(cnt, len(filenames), midi_dir))
