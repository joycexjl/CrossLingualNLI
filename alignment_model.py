import numpy as np
import fasttext.util

fasttext.util.download_model('en', if_exists='ignore') # Download the english pre-trained word vector
ft_en = fasttext.load_model('cc.en.300.bin')

fasttext.util.download_model('zh', if_exists='ignore')
ft_zh = fasttext.load_model('cc.zh.300.bin')

class FastTextAlignmentModel:
    def __init__(self, source, target):
        # Load the FastText models for English and Chinese
        self.ft_sr = fasttext.load_model(source)
        self.ft_tr = fasttext.load_model(target)

        # Load the alignment matrices (to be implemented)
        self.alignment_matrix_en = self.load_alignment_matrix()
        self.alignment_matrix_zh = self.load_alignment_matrix(ft_zh)

    def load_alignment_matrix(self, alignment_path):
        # Load the alignment matrix from the given path
        alignment_matrix = np.load(alignment_path)
        return alignment_matrix

    def align_embeddings(self, embeddings_en, embeddings_zh):
        # Apply the alignment matrices to the embeddings
        aligned_en = np.dot(embeddings_en, self.alignment_matrix_en)
        aligned_zh = np.dot(embeddings_zh, self.alignment_matrix_zh)

        return aligned_en, aligned_zh
