from transformers import BertTokenizer
from awesome_align import BertAligner

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