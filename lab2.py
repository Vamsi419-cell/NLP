import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import UnigramTagger, BigramTagger
from nltk.corpus import treebank
nltk.download('punkt')
nltk.download('treebank')
sentences = treebank.tagged_sents()
print(sentences)

