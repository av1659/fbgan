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

from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
from utils.bio_utils import *
import os
import shutil

class PsipredAnalyzer():
    def __init__(self, run_name='test'):
        self.tmp_dir = './tmp/' + run_name + '/'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.num_cpu = 10

    def evaluate_model(self):
        print("Psipred analyzer not evaluating on test set")
        return 0,0

    def parseOutput(self, out_file, struc='H'):
        '''
        Return counts of secondary structure (struc; default 'H' for alpha helix) in given file
        '''
        with open(out_file, 'r') as f:
            secstruc = f.read().splitlines()
        return secstruc.count(struc)

    def predict_model(self, input_seqs):
        #put out temp fasta sequences
        commands, out_files = [],[]
        for i,seq in enumerate(input_seqs):
            try:
                prot_seq = geneToProtein([seq], True)[0]
            except:
                print("Error in sequence {}".format(i))
                continue
            filename = self.tmp_dir + 'input_seqs_{}'.format(i)
            out_file = self.tmp_dir + 'output_seqs_{}.out'.format(i)
            with open(filename+'.fasta', 'w') as f:
                f.write('>input_seq_{}\n{}'.format(i, prot_seq))
            commands += ["~/psipred/psipred/runpsipred_single_outDir {} {}; cat {} | awk -F ' ' '{{print $3}}' > {}".format(\
                filename + ".fasta", self.tmp_dir, filename + ".ss2", out_file)]
            out_files += [out_file]
        pool = Pool(self.num_cpu)
        for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
            if returncode != 0:
               print("%d command failed: %d" % (i, returncode))
        #parse output files and put in the predictions
        all_preds = np.zeros((len(input_seqs),))
        for out_file in out_files:
            file_idx = out_file.split('_')[-1]
            file_idx = int(file_idx.split('.')[0])
            all_preds[file_idx] = self.parseOutput(out_file)
            os.remove(out_file)
        return all_preds

def main():
    test_str = 'ATGGTGATGCTGCTCATGTTCCGAAAGCTCCTGTTCACGCGCTTGCCACTCGTGGTGGTCCTCACACACGTCTTGCTGAGGCTCCTCACGCTTGAGGTTGTACTGGTGGTCCACATGCTGATCTTCGGACTTTTCCACGTTGCTTGCTTGCGCTAA'
    analyzer = PsipredAnalyzer()
    print(analyzer.predict_model([test_str]))

if __name__ == '__main__':
    main()
