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

import argparse
from collections import defaultdict
import numpy as np

codon_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

aa_table = defaultdict(list)
for key,value in codon_table.items():
    aa_table[value].append(key)

def readFasta(filename):
    try:
        f = file(filename)
    except IOError:
        print("The file, %s, does not exist" % filename)
        return

    order = []
    sequences = {}
    for line in f:
        if line.startswith('>'):
            name = line[1:].rstrip('\n')
            name = name.replace('_', ' ')
            order.append(name)
            sequences[name] = ''
        else:
            sequences[name] += line.rstrip('\n').rstrip('*')
    print("%d sequences found" % len(order))
    return order, sequences

def geneToProtein(dna_seqs, verbose=True):
    global codon_table
    p_seqs = []
    for dna_seq in dna_seqs:
        p_seq = ""
        if dna_seq[0:3] != 'ATG':
            if verbose: print("Not valid gene (no ATG)")
            continue
        for i in range(3, len(dna_seq), 3):
            codon = dna_seq[i:i+3]
            try:
                aa = codon_table[codon]
                p_seq += aa
                if aa == '_': break
            except:
                if verbose: print("Error! Invalid Codon {} in {}".format(codon, dna_seq))
                break
        if len(p_seq) <= 2: #needs to have stop codon and be of length greater than 2
            if verbose: print("Error! Protein too short.")
        elif p_seq[-1] != '_':
            if verbose: print("Error! No valid stop codon.")
        else:
            p_seqs += [p_seq[:-1]]
    return p_seqs

def proteinToDNA(protein_seqs):
    global aa_table
    stop_codons = ['TAA', 'TAG', 'TGA']
    dna_seqs = []
    for p_seq in protein_seqs:
        dna_seq = [np.random.choice(aa_table[aa]) for aa in p_seq]
        stop_codon = np.random.choice(stop_codons)
        dna_seqs += ['ATG' + "".join(dna_seq)+ stop_codon]
    return dna_seqs

def main():
    parser = argparse.ArgumentParser(description='protein to dna.')
    parser.add_argument("--dataset", default="random", help="Dataset to load (else random)")
    args = parser.parse_args()
    outfile = './samples/' + args.dataset + '_dna_seqs.fa'
    with open(args.dataset,'rb') as f:
        dna_seqs = f.readlines()
    p_seqs = geneToProtein(dna_seqs)

if __name__ == '__main__':
    main()
