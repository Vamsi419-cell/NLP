import nltk
import re
from nltk import CFG,RegexpParser
from nltk.parse.generate import generate
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
text= " todays date is 12/3/6 and $500 dollars"
date =re.findall(r'\b \d{1,2}/\d{1,2}/\d{1,2}\b',text)
money = re.findall(r'[$]\d+',text)

print(date)
print(money)
#2nd generating sents from grammer
g =CFG.fromstring("""
                  S -> NP VP
                  NP ->'I'|'You'
                  VP ->'eat'|'run'
                  
                  """)
for s in generate(g,n=5):
    print(' '.join(s))
#3rd  Regexp parser
text = "The big dog chased the cat"
tokens = word_tokenize(text)
pos_tagged = nltk.pos_tag(tokens)
grammer = "NP : {<DT><JJ><NN>}"
reg_exp = RegexpParser(grammer)
print(reg_exp.parse(pos_tagged))


