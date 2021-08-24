import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords # first nltk.download("stopwords")
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import FreqDist

# Your text 
mytext = ("This is where I need to put my text")

# Tokenize
mytext_sentences = sent_tokenize(mytext) # this will return the sentences in your text
mytext_words = word_tokenize(mytext) # this will return the tokens of your text

# Stop words 
stop_words = set(stopwords.words("english")) # create the stop words set
real_words = [] # create an empty list for all the words that are not stop words

for word in mytext:
	if word.casefold() not in stop_words: # this will check if a token is in the stop_words set, caseload() to search both UPPERCASE and lowercase 
		real_words.append(word) # this will append the token that is a word in your list real_words

# The Pythonic way to do it is by using list comprehension
real_words = [
	word for word in my_text if word.casefold() not in stop_words
]		

# Stemming
stemmer = SnowballStemmer(language='english') # this is also called Porter2 because it is an updated version of the populer PorterStemmer
stemmed_words = [stemmer.stem(word) for word in real_words] # list comprehension. Remeber to ALWAYS tokenize your text before doing Stemming 

# Lemmatizing
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in mytext_words]

# POS tagging (parts of speech)
# JJ (adjectives), NN (nouns), RB (adverbs), PRP (pronuns), VB (verbs_)
text_pos_tag = nltk.pos_tag(mytext_words) # use nltk.pos_tag and the list of your tokens

# Chunking (phrases)
my_regex = "NP: {<DT>?<JJ>*<NN>}" # create a regular expression
chunk_parser = nltk.RegexpParser(my_regex) # create your chunk parser using your regular expression
tree = chunk_parser.parse(text_pos_tag)
tree.draw() # will print the parse tree

# Chinking (is the opposite of Chunking)
my_regex_chinking = """
... Chunk: {<.*>+}
...        }<JJ>{"""

# { you want to include} vs } you want to exclude {
chunck_parser = nltk.RegexpParser(my_regex_chinking)
tree = chunk_parser.parse(text_pos_tag)
tree.draw()

# NER (Named Entity Recognition)
nltk.download("maxent_ne_chunker")
nltk.download("words")
ner_tree = nltk.ne_chunck(text_pos_tag)
ner_tree.draw() # this will print the words and they NER label

def extract_ne(my_text): # create a function to extract named entities
	words = word_tokenize(my_text, language=language)
	tags = nltk.pos_tag(words)
	tree = nltk.ne_chunk(tags)
	return set(
		" ".join(i[0] for i in t)
		for t in tree if hasattr(t, "label") and t.label() == "NE"
		)

# Concordance
my_text.concordance("Eyes") # this will print the occurences of the world "eyes" on my text

# Dispersion plot (how much a particular word appears and where it appears)
my_text.dispersion_plot(
	["eyes", "head", "kiss"]
	)

# Frequency distribution (the most popular words). Better without stop words
frequency_distribution = FreqDist(my_text)
frequency_distribution.most_common(25) # will return the 25 most popular words in your text
frequency_distribution.plot(20, cumulative=True) # you can plot your results

# Collocations (words that are used together)
my_text.collocations()

lemmatized_words = [lemmatizer.lemmatize(word) for word in my_text] # always lemmatize your text beforehand
new_text = nltk.Text(lemmatized_words) # pass the lemmatized words into an NLTK text
new_text.collocations()


# standard Corpus in NLTK
nltk.download("book")	
from nltk.book import * 