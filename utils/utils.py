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

import numpy as np
import matplotlib.pyplot as plt
import time, math
plt.switch_backend('agg')

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def splitTrainTestValLists(dataset, percent_train, percent_val = 0):
    num_train = int(percent_train*len(dataset))
    num_val = int(percent_val*len(dataset))
    train = dataset[:num_train]
    val = dataset[num_train:(num_train + num_val)]
    test = dataset[(num_train + num_val):]
    if num_val == 0: return train, test
    return train, val, test

def plot_losses(losses_list, legends_list, file_out):
    assert len(losses_list) == len(legends_list)
    for i, loss in enumerate(losses_list):
        plt.plot(loss, label=legends_list[i])
    plt.legend()
    plt.savefig(file_out)
    plt.close()

def decode_one_seq(img, letter_dict = {'A':0, 'C':1, 'G':2, 'T':3}):
    seq = ''
    for row in range(len(img)):
        on = np.argmax(img[row,:])
        seq += letter_dict[on]
    return seq

def splitTrainTestVal(dataset, percent_train, percent_val = 0):
    num_train = int(percent_train*len(dataset))
    num_val = int(percent_val*len(dataset))
    train = dataset[:num_train, :]
    val = dataset[num_train:(num_train + num_val), :]
    test = dataset[(num_train + num_val):, :]
    if num_val == 0: return train, test
    return train, val, test

def one_hot_encode(seqs, letter_dict = {'A':0, 'C':1, 'G':2, 'T':3}):
    seqs_nums = np.array([[letter_dict[base] for base in seq.rstrip()] for seq in seqs], dtype=np.float32)
    seqs_one_hot = (np.arange(4) == seqs_nums[...,None]).astype(np.float32)
    #seqs_one_hot = np.expand_dims(seqs_one_hot, axis=3)
    seqs_one_hot = np.swapaxes(seqs_one_hot, 1, 2)
    return seqs_one_hot

def findMotif(seq, motif):
    if motif in seq:
        return True
    return False
