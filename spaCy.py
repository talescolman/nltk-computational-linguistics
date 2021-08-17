import spacy 
from spacy.matcher import Matcher, PhraseMatcher # to import the Matcher
from spacy.lang.en import English # to load the English language
from spacy.tokens import Doc, Span # Doc to create your own Doc, Span to use the Span


# load the English-language package, go to SpaCY to find other languages
nlp = spacy.load("en_core_web_sm") 

# create your input, the nlp object will tokenize and do all ther work for you
doc = nlp("This is my spaCy tutorial!")

# iterate your input tokens
for token in doc:
    print(token.text, token.pos_, token.lemma_, 
    	token.tag_, token.dep_, token.shape_, token.is_stop) 

# token.text return the token
# token.pos_ returns part-of-speech tag
# token.lemma_ the base form of the word
# token.tag_ returns the detailed part-of-speech tag
# token.dep_ returns Syntactic dependency
# token.shape_ returns The word shape – capitalization, punctuation, digits.
# token.is_stop returns if the token part of a stop list, i.e. the most common words of the language

# create an object for the first token
token = doc[1]
print(token.text)  # this will print the word "This"  

# create an object for more than one token
span = doc[1:3]
print(span.text) # this will print "This is my"

# tokens lexical atributes
print("Index: ", [token.i for token in doc]) # will return the index [0, 1, 2, 3, 4, 5, 6]
print("Text: ", [token.text for token in doc]) # will return the text split in tokens ["This", "is" , "my" , "spaCy" , "tutorial" , "!"]
print("is_alpha: ", [token.is_alpha for token in doc]) # returns True if the token is part of the alphabet [True, True, True, True, True, False]
print("is_punct: ", [token.is_punct for token in doc]) # returns True if the token is a punctuation [False, False, False, False, True]
print("like_num: ", [token.like_num for token in doc]) # returns True if the token is a number [False, False, False, False, False]

# Named entities (Brands, Countries, etc...)
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
	print(ent.text, ent.label)

# You can search for the meaning of most spaCy labels using:
spacy.explain("dobj") # you can change dobj to any spaCy label

# Matcher lets you find words and phrases. Rules can refer to token annotations (like the text or part-of-speech tags), as well as lexical attributes.
matcher = Matcher(nlp.vocab)

pattern = [{"TEXT": "VW"}, {"TEXT": "UP"}]
matcher.add("VW_PATTERN", None, pattern) # add an expression to the comprator

doc = nlp("I bought a new VW UP.")
matches = matcher(doc) # call the matcher using your input as attribute

for match_id, start, end in matches: # iterate the matchings
	matched_span = doc[start:end]
	print(matched_span.text)

# you can have complex patterns
pattern = [
	{"IS_DIGIT": True},
	{"LOWER": "red"},
	{"LOWER": "ford"},
	{"LOWER": "puma"},
	{"IS_PUNCT": True}
]

# searching for a VERB + NOUN
pattern = [
	{"LEMMA": "like", "POS": "VERB"}, # Verb like
	{"POS": "NOUN"} # Noun after the verb
]

# You can combine your search patterns with operators. Operator or quantifier to determine how often to match a token pattern. 
# {"OP": "!"} 0 times
# {"OP": "?"} 0 or 1 times
# {"OP": "+"} 1 or more times
# {"OP": "*"} 0 or more times

# Strings are saved as a hash value. If the same word appears 3 times in a text they are all stored as a single hash
print("hash value:", nlp.vocab.strings["new"]) # this will print the hash value of new
print("string value: ", nlp.vocab.strings[3878344210879014392]) # this will print the word "new"

# instead of import your data, you can create your own doc
words = ["He", "is", "not", "sad", "!"]
spaces = [True, True, True, False, False] # True if after the word you should have a blank space, False if not
doc = Doc(nlp.vocab, words=words, spaces=spaces) # this creates your doc

# how to use Span (nameofyourtext, i of first word, i of last word that will not be printed)
span = Span(doc, 0, 2)

# you can also create a label for your Span and then save it
span_with_label = Span(doc, 0, 2, label="Pronouns")
doc.ents = [span_with_label]

# print the text + the entity labels
print([(ent.text, ent.label_) for ent in doc.ents])

# iterate the tokens to found a proper noun before a verb
doc = nlp("London is expensive in the summer")

for token in doc: # iterate the tokens
	if token.pos_ == "PROPN": # if the token is a proper noun
		if doc[token.i + 1].pos_ == "VERB": # checks if the next token is a verb (that's why token.i + 1)
			print("Found proper noun before a verb:", token.text)

# SIMILARITY can be used with Doc.similarity(), Span.similarity or Token.similarity() and returns a value beween 0 and 1

# always use the long version of a language package to this task, like en_core_web_lg
nlp = spacy.load("en_core_web_md")

# compare two Docs
doc1 = nlp("I hate school")
doc2 = nlp("I hate university")
print(doc1.similarity(doc2)) # this will return 0.835 ... Any result bigger than 0. is relevant.

# compare two tokens
doc =nlp("I hate VW and Ford")
token1 = doc[2]
token2 = doc[4]
print(token1.similarity(token2))

# compare a doc to a token
doc = nlp("I hate Berlin")
token = nlp("Hamburg") [3]
print(doc.similarity(token))

# compare part (span) of a doc
span = nlp("I love to eat pizza in the morning") [2:5]
doc = nlp("She loves soda in the morning")
print(span.smilarity(doc))

# you can return the head token and the main token of a span

for match_id, start, end in matcher(doc)
	span = doc[start:end]
	print("Matched span:", span.text) 
	print("Root token:", span.root.text) # the main token
	print("Root head token:", span.root.head.text) # the head root token
	print("Previous token:" , doc[start - 1].text, doc[start - 1].pos_) # returns the previous Token + its part-of-speech tag

# PhraseMatcher just like Matcher but for lists, dictionaries and docs

matcher = PhraseMatcher(nlp.vocab)

pattern = nlp("VW Golf") # what you are looking for
matcher.add("CAR", none, pattern) # add what you are looking for to the matcher
doc = nlp("I have a VW Golf") # where you are looking for

for match_id, start, end in matcher(doc)
	span = doc[start:end]
	print("Matched span: ", span.text) # thi will print -> Matched span: VW Golf

# add custom options to your pipeline
def custom_component(doc): # create your custom pipeline
	print("Doc length:", len(doc)) # this will print the length of your document
	return doc

nlp.add_pipe(custom_component, last=true) # add your customization to the pipeline
# you can change the last=True to:
# last=True will add your custom to the end of the pipeline
# first=True  will add your custom to the begining of the pipeline
# before="ner" will add your custom before the EntityRecognizer
# after="tagger" will add your custom after the Tagger

# add custom attributes, on this example our code will search for the wikipedia artice of people mentioned on the text
# the documentation to add cutom attributes is long and complex, so this is just an example. In order to learn more please visit paCy's official documentation

def get_wikipedia_url(span):
	if ____ in ("PERSON", "ORG", "GPE", "LOCATION"):
		entity_text = span.text.replace(" ", "_")
		return "https://en.wikipedia.org/w/index.php?search=" + entity_text

Span.set_extension("wikipedia_url", getter=get_wikipedia_url)

doc = nlp("The Italian government, looking to contain a fresh surge in coronavirus cases, announced on Thursday that people would need to have proof of immunity to access an array of services and leisure activities.")
for ent in doc.ents:
	print(ent.text, ent._.wikipedia_url)

# if you dont need the whole NLP pipeline, only the Tokenization of any Doc, you an use:
doc = nlp.make_doc("This is my text!")

# another option is to disable some parts of your pipeline
with nlp.disable_pipes("tagger", "parser"):
	doc = nlp(text)
	print(doc.ents)

# To train your own model we will use another tutorial
