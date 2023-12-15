from transformers import BertTokenizer
import fasttext.util

fasttext.util.download_model('en', if_exists='ignore') # Download the english pre-trained word vector
ft_en = fasttext.load_model('cc.en.300.bin')

fasttext.util.download_model('zh', if_exists='ignore')
ft_zh = fasttext.load_model('cc.zh.300.bin')

def get_ft_embeddings(ft_model, sentence):
    words = sentence.split()  # Simple tokenization, might need adjustments for Chinese
    embeddings = [ft_model.get_word_vector(word) for word in words]
    return embeddings

# class FastTextAlignmentModel:
#     def __init__(self, model_path_en, model_path_zh):
#         # Load the FastText models for English and Chinese
#         self.ft_en = fasttext.load_model(model_path_en)
#         self.ft_zh = fasttext.load_model(model_path_zh)

#     def get_embeddings(self, sentence, language):
#         if language == 'en':
#             model = self.ft_en
#         elif language == 'zh':
#             model = self.ft_zh
#         else:
#             raise ValueError("Unsupported language")
        
#         # Tokenize the sentence and get embeddings
#         # Note: For Chinese, a proper tokenizer should be used
#         words = sentence.split() 
#         embeddings = np.array([model.get_word_vector(w) for w in words])

#         return embeddings

#     def align_embeddings(self, embeddings_en, embeddings_zh):
#         # Placeholder for alignment logic
#         # Assuming embeddings_en and embeddings_zh are the embeddings of two aligned sentences

#         # In the real scenario, apply the transformation learned from MUSE here
#         # For now, we'll return the embeddings as is, since we haven't run MUSE

#         return embeddings_en, embeddings_zh


class WordAlignmentModel:
    def __init__(self, source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.aligner = BertAligner("bert-base-multilingual-cased", token_type="bpe")

    def tokenize(self, text, lang):
        # Add specific tokenization logic for different languages if needed
        return self.tokenizer.tokenize(text)

    def align_sentences(self, sentence_source, sentence_target):
        tokenized_source = self.tokenize(sentence_source, self.source_lang)
        tokenized_target = self.tokenize(sentence_target, self.target_lang)

        # Perform alignment
        alignments = self.aligner.get_word_aligns([tokenized_source], [tokenized_target])
        return alignments