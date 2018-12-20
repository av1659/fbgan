#    Copyright (C) 2018 Anvita Gupta
#
#    This program is free software: you can redistribute it and/or  modify
#    it under the terms of the GNU Affero General Public License, version 3,
#    as published by the Free Software Foundation.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from utils import *

class Lang:
    def __init__(self, name, tokenizerFunc=lambda x: x.strip()):
        self.name = name
        self.tokenizerFunc = tokenizerFunc
        self.token2index = {"P":0}
        self.token2count = {}
        self.index2token = {0:"P"} #pad character
        self.n_tokens = 1 # Count SOS and EOS
        self.PAD_token = 0
        self.SOS_token = -1
        self.EOS_token = -1

    def index_string(self, sentence):
        for token in self.tokenizerFunc(sentence):
            self.index_token(token)

    def index_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1

    def preprocess_seqs(self, sequences):
        '''
        Tokenize and pad sequences to same length (everything but one hot encoding)
        '''
        tokenized = [[self.token2index[c] for c in self.tokenizerFunc(l)] for l in sequences]
        padded = self.pad_seqs(tokenized)
        return np.array(padded,dtype='int32')

    def pad_seqs(self, sequences):
        '''
        Pad sequences to longest length(default) or given length
        Can be sequences of TOKENS as well!
        '''
        length = max([len(seq) for seq in sequences])
        padded_seqs = []
        for seq in sequences:
            padded_seq = seq + [self.PAD_token]*(length - len(seq))
            padded_seqs += [padded_seq]
        return padded_seqs

def pad_seq(seq, pad_char, length):
    padded_seq = seq + [pad_char]*(length - len(seq))
    return padded_seq
