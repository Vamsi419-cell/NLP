import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')
text ="NLP is very good . Ai is a very good subject"
print(f"sentences are  {sent_tokenize(text)}")
words = [w.lower() for w in word_tokenize(text)]
freq = FreqDist(words)
for word, count in freq.items():
    print(f"{word} : {count}")
#2nd question pos tagging
nltk.download('averaged_perceptron_tagger')
pos_tags =nltk.pos_tag(words)
print(pos_tags)
#3rd question web page tokens & freq clock
import re , requests
url="https://www.google.com"
text = requests.get(url).text
tokens = re.findall(r'[A-za-z0-9]+',text.lower())
freq = FreqDist(tokens)
print(freq.most_common(10))
print(freq.plot(10))
#4rd porter stemmer and stop words
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
text = "I am learning NLP and I love learning Python in NLP"
token = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
filtered = [w for w in token if w.isalnum() and w not in stop_words]
ps = PorterStemmer()
stemmed_words = [ps.stem(w) for w in filtered]
print(stemmed_words)