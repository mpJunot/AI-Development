import numpy as np
from core.lstm import LSTM

class Encoder:
    def __init__(self, input_size, hidden_size):
        self.rnn = LSTM(input_size, hidden_size)
    def forward(self, x, h0, c0=None):
        return self.rnn.forward(x, h0, c0)

class Decoder:
    def __init__(self, input_size, hidden_size, output_size):
        self.rnn = LSTM(input_size, hidden_size)
    def forward(self, x, h0, c0=None, encoder_outputs=None):
        return self.rnn.forward(x, h0, c0)

class Seq2Seq:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, tgt, h0, c0=None):
        encoder_outputs = self.encoder.forward(src, h0, c0)
        return self.decoder.forward(tgt, h0, c0, encoder_outputs)
