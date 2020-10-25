# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:33:28 2020

@author: joshc
"""
from spacy.lang.en import English
nlp = English()

# Chapter 2
text = "Hi how are you today? I'm swell!"
doc = nlp(text)
# for token in doc:
#     print(token.i, token.text)

# 1) Strings and Hashes
today_hash = doc.vocab.strings["today"]
print(today_hash)
word = doc.vocab.strings[today_hash]
print(word)

# Since hello isn't stored in the string store, it hasn't been hashed yet
today_hash = nlp.vocab.strings["Hello"]
print(today_hash)
# word = nlp.vocab.strings[today_hash]
# print(word)

# But, once you use it in a passed in sentence, the hash label is created
# Words aren't prehashed, but passed through a hashing function as needed
text = "Hello, my name is Josh"
doc = nlp(text)
today_hash = nlp.vocab.strings["Hello"]
print(today_hash)
word = nlp.vocab.strings[today_hash]
print(word)

# 4) Docs and Span Classes

# 5) Create a Doc object
# Import the Doc class
from spacy.tokens import Doc

# Desired text: "spaCy is cool!"
# Spaces list holds info about whether token is followed by a space
words = ["spaCy", "is", "cool", "!"]
spaces = [True, True, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

# 6) Creating Docs and Spans
# Import the Doc and Span classes
from spacy.tokens import Span

words = ["I", "like", "David", "Bowie"]
spaces = [True, True, True, False]

# Create a doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

# Create a span for "David Bowie" from the doc and assign it the label "PERSON"
span = Span(doc, 2, 4, label="PERSON")
print(span.text, span.label_)

# Add the span to the doc's entities
doc.ents = [span]

# Print entities' text and labels
print([(ent.text, ent.label_) for ent in doc.ents])

# 7) Efficient Use of DataStructures
""" Part 2
Rewrite the code to use the native token attributes instead of lists of 
token_texts and pos_tags. Loop over each token in the doc and check the 
token.pos_ attribute.

Use doc[token.i + 1] to check for the next token and its .pos_ attribute.
If a proper noun before a verb is found, print its token.text.
import spacy
"""
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp("Berlin looks like a nice city")

for token in doc:
    # Check if the current token is a proper noun
    if token.pos_ == "PROPN":
        # Check if the next token is a verb
        if token.i+1 < len(doc) and doc[token.i+1].pos_ == "VERB":
            result = token.text
            print("Found proper noun before a verb:", result)

# 8) Word Vectors and Similarities
import en_core_web_md
nlp2 = en_core_web_md.load()
doc = nlp2("Hello there!")
doc2 = nlp2("Hi there!")

print(doc.similarity(doc2))
print(len(doc[0].vector))

# Beware of assuming the cosine similarity imposes meaning all the time
doc3 = nlp2("I hate cats")
doc4 = nlp2("I love cats")
print(doc3.similarity(doc4))

# the similarity is 0.9409261755229907, but the sentiment expressed is 
# completely opposite

# 10) Comparing similarities (individual tokens)
doc = nlp2("TV and books")
token1, token2 = doc[0], doc[2]

# Get the similarity of the tokens "TV" and "books"
similarity = token1.similarity(token2)
print(similarity)

doc = nlp2("This was a great restaurant. Afterwards, we went to a really nice bar.")

# Create spans for "great restaurant" and "really nice bar"
span1 = doc[3:5]
span2 = doc[-4:-1]

# Get the similarity of the spans
similarity = span1.similarity(span2)
print(similarity, span1.text, span2.text)

# 11) Combining models and rules
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
doc = nlp(
    "Twitch Prime, the perks program for Amazon Prime members offering free "
    "loot, games and other benefits, is ditching one of its best features: "
    "ad-free viewing. According to an email sent out to Amazon Prime members "
    "today, ad-free viewing will no longer be included as a part of Twitch "
    "Prime for new members, beginning on September 14. However, members with "
    "existing annual subscriptions will be able to continue to enjoy ad-free "
    "viewing until their subscription comes up for renewal. Those with "
    "monthly subscriptions will have access to ad-free viewing until October 15."
)

# Original Incorrect patterns
pattern1 = [{"LOWER": "Amazon"}, {"IS_TITLE": True, "POS": "PROPN"}]
pattern2 = [{"LOWER": "ad-free"}, {"POS": "NOUN"}]

# Corrected match patterns
pattern1 = [{"LOWER": "amazon"}, {"IS_TITLE": True, "POS": "PROPN"}]
pattern2 = [{"LOWER": "ad"}, {"TEXT": "-"}, {"LOWER": "free"}, {"POS": "NOUN"}]

# Initialize the Matcher and add the patterns
matcher = Matcher(nlp.vocab)
matcher.add("PATTERN1", None, pattern1)
matcher.add("PATTERN2", None, pattern2)

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Print pattern string name and text of matched span
    print(doc.vocab.strings[match_id], doc[start:end].text)


# 14) Efficient matching (file import is broken because was online code)
# import json
# with open("exercises/en/countries.json") as f:
#     COUNTRIES = json.loads(f.read())

# nlp = English()
# doc = nlp("Czech Republic may help Slovakia protect its airspace")

# # Import the PhraseMatcher and initialize it
# from spacy.matcher import PhraseMatcher

# matcher = PhraseMatcher(nlp.vocab)

# # Create pattern Doc objects and add them to the matcher
# # This is the faster version of: [nlp(country) for country in COUNTRIES]
# patterns = list(nlp.pipe(COUNTRIES))
# matcher.add("COUNTRY", None, *patterns)

# # Call the matcher on the test document and print the result
# matches = matcher(doc)
# print([doc[start:end] for match_id, start, end in matches])


# 15) Extract Countries and Relationships
# File imports from online source

# import spacy
# from spacy.matcher import PhraseMatcher
# from spacy.tokens import Span
# import json

# with open("exercises/en/countries.json") as f:
#     COUNTRIES = json.loads(f.read())
# with open("exercises/en/country_text.txt") as f:
#     TEXT = f.read()

# nlp = spacy.load("en_core_web_sm")
# matcher = PhraseMatcher(nlp.vocab)
# patterns = list(nlp.pipe(COUNTRIES))
# matcher.add("COUNTRY", None, *patterns)

# # Create a doc and reset existing entities
# doc = nlp(TEXT)
# doc.ents = []

# # Iterate over the matches
# for match_id, start, end in matcher(doc):
#     # Create a Span with the label for "GPE"
#     span = Span(doc, start, end, label="GPE")

#     # Overwrite the doc.ents and add the span
#     doc.ents = list(doc.ents) + [span]

#     # Get the span's root head token
#     span_root_head = span.root.head
#     # Print the text of the span root's head token and the span text
#     print(span_root_head.text, "-->", span.text)

# # Print the entities in the document
# print([(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "GPE"])







        