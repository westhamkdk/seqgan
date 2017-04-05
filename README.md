# SeqGAN

## Requirements: 
* Tensorflow (r1.0)
* Cuda (7.5+)
* nltk python package
* sugartensor

## Introduction
Apply Generative Adversarial Nets to generating sequences of discrete tokens.

![](https://github.com/LantaoYu/SeqGAN/blob/master/figures/seqgan.png)

The illustration of SeqGAN. Left: D is trained over the real data and the generated data by G. Right: G is trained by policy gradient where the final reward signal is provided by D and is passed back to the intermediate action value via Monte Carlo search.  

The research paper [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](http://arxiv.org/abs/1609.05473) has been accepted at the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17).




## Usage

###1. Data

 To build input data, place midi files inside midi/ folder and run midi_io.py. It will create note_mapping_dict.pkl and SeqGAN_sugartensor/save/midi_trans.pkl.
 Unless you remove SeqGAN_sugartensor/save/midi_trans.pkl, it won't generate any files.

###2. Training
Get inside SeqGAN_sugartensor and run

```
python seq_gan.py
```

It will start pretraining. (Policy gradient step will be added)

```
LSTM_graph.py
```

refers to generator at pretraining

and
```
CNN_graph.py
```
refers to discriminator.


###3. Outputs

Conversion is needed for playing. Run trans_generated_to_midi(path) function inside midi_io.py
