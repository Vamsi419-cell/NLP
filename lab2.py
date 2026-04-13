import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import UnigramTagger, BigramTagger,RegexpTagger
from nltk.corpus import treebank
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.tag.brill import brill24
nltk.download('punkt')
nltk.download('treebank')
sentences = treebank.tagged_sents()
training = sentences[:3000]
testing=sentences[3000:]
unigram = UnigramTagger(training)
bigram = BigramTagger(training,backoff=unigram)
print(f"Bigramtagger {bigram.accuracy(testing)}")
print(f"UnigramTagger{unigram.accuracy(testing)}")
templete = brill24()
trainer = BrillTaggerTrainer(unigram,templates=templete)
brill_tagger = trainer.train(training)
print(f"Brillgram {brill_tagger.accuracy(testing)} ")
pattern = [
    (r'.*ing$','VBG'),
    (r'.*ed','VBD'),
    (r'.*','NN')
]
text ="John is learning NLP and played cricket"
tokens = word_tokenize(text.lower())
reg_ex = RegexpTagger(pattern)
print(f"Regexptagger {reg_ex.tag(tokens)}" )


