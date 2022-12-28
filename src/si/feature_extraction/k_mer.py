import re
from si.data.dataset import Dataset
import itertools
import numpy as np

class KMer:
    """
    Determines the frequency of all possible K-mers of a given size ('k' parameter)
    for each DNA or protein sequence present in a given Dataset instance.
    """

    def __init__(self, k:int = 3, seq_type:str = "dna"):
        """
        Stores variables.

        Parameters
        ----------
        :param k: The length of each possible K-mer to determine
        :param seq_type: The sequence type ("dna" or "protein"), to define the alphabet to use
        """
        assert seq_type.lower() in ["dna", "protein"], "Choose a valid sequence type ('dna' or 'protein')."

        if seq_type.lower() == "dna":
            self.alpha = "ACGT"
        else:
            self.alpha = "ACDEFGHIKLMNPQRSTVWY"

        self.k = int(k)



    def fit(self, dataset:Dataset) -> 'KMer':
        """
        Determines all possible K-mers of a specified size ('k' parameter).

        Parameters
        ----------
        :param dataset: An instance of the Dataset class (required for consistency purposes only)
        """
        self.kmers = ["".join(kmer) for kmer in itertools.product(self.alpha, repeat=self.k)]
        return self


    def _get_seq_kmers(self, seq:str) -> dict:
        """
        Auxiliary function to determine the input sequence's K-mer frequencies

        Parameters
        ----------
        :param seq: A DNA/Protein sequence
        """
        seq_kmers = set(re.findall(fr"(?=({'.' * self.k}))", seq))
        kmer_freq = {kmer: 0 for kmer in self.kmers}
        seq_length_k = len(seq) - self.k + 1 #Number of total K-mers in the sequence

        for kmer in seq_kmers:
            kmer_num = len(re.findall(fr"(?=({kmer}))", seq)) #Number of a specific K-mer in the sequence
            kmer_freq[kmer] = kmer_num / seq_length_k         #Frequency of a specific K-mer

        return kmer_freq


    def transform(self, dataset:Dataset) -> 'Dataset':
        """
        Determines the frequency of each K-mer for all sequences in the given dataset.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class containing sequences
        """
        result_freqs = {kmer:[] for kmer in self.kmers}
        for seq in dataset.X:
            kmer_freq = self._get_seq_kmers(*seq)
            result_freqs = {k: [*result_freqs[k], kmer_freq[k]] for k in kmer_freq.keys()}

        x = np.array(list(result_freqs.values())).T
        return Dataset(x, dataset.y, features=self.kmers, label=dataset.label)


    def fit_transform(self, dataset:Dataset) -> 'Dataset':
        """
        Runs the fit() and transform() methods.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class containing sequences
        """
        self.fit(dataset)
        return self.transform(dataset)

