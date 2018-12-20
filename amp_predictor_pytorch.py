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

import random, os, h5py, math, time, glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from utils.utils import *
from utils.bio_utils import *
from utils.lang_utils import *

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, batch_size, hidden_dim):
        super(GRUClassifier, self).__init__()
        self.hidden = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=False, num_layers=2, dropout=0.3)
        self.linear = nn.Linear(hidden_dim, 1) # input dim is 64*2 because its bidirectional
        self.batch_size = batch_size
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x, h):
        x = self.embedding(x)
        x, h = self.gru(x, h)
        x = F.sigmoid(self.linear(x[-1])) # sigmoid output for binary classification
        return x, h

    def init_hidden(self):
        if self.use_cuda:
            return Variable(torch.randn(2, self.batch_size, self.hidden)).cuda()
        return Variable(torch.randn(2, self.batch_size, self.hidden))

def indexes_from_sentence(lang, sentence):
    return [lang.token2index[t] for t in sentence]

class ACPClassifier():
    def __init__(self, hidden_dim=128, batch_size=64, learning_rate=0.001, epochs=50,
        dataset='./data/AMP_dataset.fa', run_name='class_pytorch_drop_03'):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_epochs = epochs
        self.learning_rate = learning_rate
        self.use_gpu = True if torch.cuda.is_available() else False
        self.pairs = self.load_data(dataset)
        self.train_pairs, self.val_pairs, self.test_pairs = splitTrainTestValLists(self.pairs, 0.6, 0.2)
        print( "{} Training Pairs; {} Validation Pairs".format(len(self.pairs), len(self.val_pairs)))
        self.build_model()
        self.checkpoint_dir = './checkpoint/' + run_name + '/'
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        self.init_epoch = self.load_model()

    def load_data(self, dataset):
        pairs = []
        self.lang = Lang("dna")
        with open(dataset, 'r') as f:
            for line in f:
                seq, label = line.split()
                self.lang.index_string(seq)
                pairs += [(seq, int(label))]
        np.random.shuffle(pairs)
        return pairs

    def build_model(self):
        self.rnn = GRUClassifier(self.lang.n_tokens, self.batch_size, hidden_dim=128)
        if self.use_gpu:
            self.rnn.cuda()
        self.optimizer = optim.Adam(self.rnn.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

    def save_model(self, epoch):
        torch.save(self.rnn.state_dict(), self.checkpoint_dir + "model_weights_{}.pth".format(epoch))

    def load_model(self):
        '''
            Load model parameters from most recent epoch
        '''
        list_model = glob.glob(self.checkpoint_dir + "model*.pth")
        if len(list_model) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        chk_file = max(list_model, key=os.path.getctime)
        epoch_found = int( (chk_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found!".format(epoch_found))
        self.rnn.load_state_dict(torch.load(chk_file))
        return epoch_found

    def train_model(self):
        num_batches = int(len(self.train_pairs)/self.batch_size)
        start = time.time()
        print_loss_total, total_acc, total_overall = 0, 0, 0
        min_val_loss = 10000
        print( "Starting training...")
        train_loss_f = open(self.checkpoint_dir + "losses.txt",'a+')
        val_loss_f = open(self.checkpoint_dir + "val_losses.txt",'a+')
        counter = 0
        h = self.rnn.init_hidden()
        for epoch in range(self.init_epoch,self.n_epochs+1):
            for batch in range(num_batches):
                counter += 1
                input_batches, input_lengths, target = self.random_batch(self.train_pairs)
                target = Variable(target).type(torch.FloatTensor)
                target = target.view(self.batch_size, 1)
                if self.use_gpu: target = target.cuda()
                h.detach_()
                y_pred, h = self.rnn(input_batches, h)
                self.optimizer.zero_grad()
                loss = self.criterion(y_pred, target)
                loss.backward()
                self.optimizer.step()

                trn_preds = torch.round(y_pred.data)
                correct = torch.sum(trn_preds == target.data)
                print_loss_total += loss.data[0]
                total_acc += correct
                total_overall += self.batch_size
            val_loss, val_acc = self.evaluate_model()
            print_summary = '%s (%d %d%%) Train Loss-%.4f Train Acc- %.4f Val Loss- %.4f Val Acc-%.4f'\
                % (time_since(start, float(epoch) / self.n_epochs), epoch, float(epoch) / self.n_epochs * 100,
                print_loss_total / num_batches, float(total_acc)/total_overall, val_loss, val_acc)
            print(print_summary)
            train_loss_f.write("Epoch: {} \t Loss: {}\n Accuracy: {}\n".format(epoch, print_loss_total / num_batches, float(total_acc)/total_overall))
            val_loss_f.write("Epoch: {} \t Val Loss: {} Val Acc: {}\n".format(epoch, val_loss, val_acc))
            if val_loss < min_val_loss:
                self.save_model(epoch)
                min_val_loss = val_loss
                print("Saved model at epoch {}\n".format(epoch))
            print_loss_total, total_acc, total_overall = 0, 0, 0
        test_loss, test_acc = self.evaluate_model(validation=False)
        print("Test Loss:{}, Test Accuracy: {}\n".format(test_loss, test_acc))

    def evaluate_model(self, validation=True):
        if validation: 
            pairs = self.val_pairs
        else: 
            pairs = self.test_pairs
            print("Test Set...")
        self.rnn.train(False)
        total_loss = 0
        num_batches = int(len(pairs)/self.batch_size)
        y_scores_all, y_pred_all = np.zeros((num_batches*self.batch_size,1)), np.zeros((num_batches*self.batch_size,1))
        target_all = np.zeros((num_batches*self.batch_size,1))
        hid = self.rnn.init_hidden()
        for batch in range(num_batches):
            start_idx = batch*self.batch_size
            input_batches, input_lengths, target = self.sequential_batch(pairs, start_idx)
            target = Variable(target).type(torch.FloatTensor)
            target = target.view(self.batch_size, 1)
            if self.use_gpu: target = target.cuda()
            y_pred, hid = self.rnn(input_batches, hid)
            loss = self.criterion(y_pred, target)
            total_loss += loss.data[0]
            y_scores_all[start_idx:(start_idx+self.batch_size)] = y_pred.data
            target_all[start_idx:(start_idx+self.batch_size)] = target.data
            y_pred_all[start_idx:(start_idx+self.batch_size)] = torch.round(y_pred.data)
        self.rnn.train(True)
        fpr, tpr, thresholds = metrics.roc_curve(target_all, y_scores_all)
        auc = metrics.auc(fpr, tpr)
        prec = metrics.precision_score(target_all, y_pred_all)
        recall = metrics.recall_score(target_all, y_pred_all)
        accuracy = metrics.accuracy_score(target_all, y_pred_all)
        print("AUC: {}, Precision: {}, Recall: {}".format(auc, prec, recall))
        return total_loss/num_batches, accuracy

    def predict_model(self, input_seqs):
        pos_seqs = []
        hid = self.rnn.init_hidden()
        num_pred_batches = int(len(input_seqs)/self.batch_size)
        all_preds = np.zeros((num_pred_batches*self.batch_size, 1))
        for idx in range(num_pred_batches):
            batch_seqs = input_seqs[idx*self.batch_size:(idx+1)*self.batch_size]
            tokenized_seqs = [indexes_from_sentence(self.lang, s.strip()) for s in batch_seqs]
            input_lengths = [len(s) for s in tokenized_seqs]
            input_padded = [pad_seq(s, self.lang.PAD_token, max(input_lengths)) for s in tokenized_seqs]
            input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
            input_var = input_var.cuda() if self.use_gpu else input_var
            y_pred, hid = self.rnn(input_var, hid)
            print( "Made predictions...")
            all_preds[idx*self.batch_size:(idx+1)*self.batch_size,:] = y_pred.data.cpu().numpy()
        return all_preds

    def sequential_batch(self, pairs, start_idx):
        batch_pairs = pairs[start_idx:(start_idx + self.batch_size)]
        seqs, labels = zip(*batch_pairs)
        input_seqs = [indexes_from_sentence(self.lang, seq) for seq in seqs]
        seq_pairs = sorted(zip(input_seqs, labels), key=lambda p: len(p[0]), reverse=True)
        input_seqs, labels = zip(*seq_pairs)
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(s, self.lang.PAD_token, max(input_lengths)) for s in input_seqs]
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target = torch.LongTensor(labels)
        if self.use_gpu:
            input_var = input_var.cuda()
            target = target.cuda()
        return input_var, input_lengths, target

    def random_batch(self, pairs):
        input_seqs, labels = [],[]
        for i in range(self.batch_size):
            seq, label = random.choice(pairs)
            input_seqs.append(indexes_from_sentence(self.lang, seq))
            labels.append(label)
        seq_pairs = sorted(zip(input_seqs, labels), key=lambda p: len(p[0]), reverse=True)
        input_seqs, labels = zip(*seq_pairs)
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(s, self.lang.PAD_token, max(input_lengths)) for s in input_seqs]
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target = torch.LongTensor(labels)
        if self.use_gpu:
            input_var = input_var.cuda()
            target = target.cuda()
        return input_var, input_lengths, target

def main():
    parser = argparse.ArgumentParser(description='RNN Predictor of Antimicrobial Activity of Gene Products')
    parser.add_argument("--run_name", default='class_pytorch_drop_03', help="Name for checkpoints")
    args = parser.parse_args()
    rnn = ACPClassifier(run_name=args.run_name)
    rnn.train_model()

if __name__ == '__main__':
    main()
