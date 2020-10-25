# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:50:21 2020

@author: joshc
"""

"""
What do i want to do, and what do i need to do it.

Step 1:
    Familiazrize self with spacy
    SpaCy Course: https://course.spacy.io/en/
        
"""
from spacy.lang.en import English
###################
# Chapter 1: 
    
# 1) Intro to spaCy (Video)
print("\n-------------------")
nlp = English()
doc = nlp("Babies first document string!")
for token in doc:
    print(token.text)
    
word2 = doc[1]
print(word2.text)

print("-------------------")
# 2) Span object is a slice of the doc without extra data
span = doc[1:3]
print(span.text)

# Token attributes
print("Index:   ", [token.i for token in doc])
print("Text:   ", [token.text for token in doc])
print("is_alpha:   ", [token.is_alpha for token in doc])
print("is_punct:   ", [token.is_punct for token in doc])

# 4) Lexical Attributes
print("\n-------------------")
doc2 = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are.")
print("Text:   ", [token.text for token in doc2])
print("like_num:   ", [token.like_num for token in doc2])

for token in doc2:
    if token.like_num:
        next_token = doc2[token.i+1]
        if next_token.text == "%":
            print("Found a percent:", token.text)
                        
# 5) Statistical Models (Video)
"""
Can tag speech as part of speech, syntax, named entities
Part of Speech: token.pos_
Syntactic Dependencies: token.dep_
Named Object Prediction: ent.label_

*** Attributes in spacy that return strings usually end in an underscore

Trained on labeled example text
Can import trained models using spacy.load
Model:  en_core_web_sm (trained on web text)

spacy.explain("something") will return an explaination of function
"""
print("\n-------------------")
import spacy
import en_core_web_sm
nlp= en_core_web_sm.load()

# Predicting part of speech, syntactic dependency tags
doc = nlp("She ate the pizza before running to her car.")
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)
    
print("-------------------")
# Named object Recognition
doc = nlp("Ron said 'Look Harry, in Britain, a wizard always uses Google'")
for ent in doc.ents:
    print(ent.text, ent.label_)
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)

# Spacy helper function for common labels and tags
print(spacy.explain("GPE"))

# 9) Dealing with incorrect tokens
print("\n-------------------")

text = "Upcoming iPhone X release date leaked as Apple reveals pre-orders"

# Process the text
doc = nlp(text)

# Iterate over the entities
for ent in doc.ents:
    print(ent.text, ent.label_)

# Get the span for "iPhone X"
iphone_x = doc[1:3]

# Print the span text
print("Missing entity:", iphone_x.text)

# 10) Rule Based Matching (Video)
"""
Spacy Matcher works in a much more granular manner than regEx.

Can match across doc objects, not just strings
This allows it to match tokens, and token attributes

Can finetune searching, ie find duck(verb) vs duck(noun)

A match pattern is a dictionary organizes by {attribute: text}
Find text "iPhone X" --> [{"Text": "iPhone"}, {"Text": "X"}]

***Pattern is case sensitive, unless you use the attribute "LOWER"

import via spacy.matcher
"""
print("-------------------")

from spacy.matcher import Matcher
# Load the model (already stored in nlp from section 5 above)

# Initialize the matcher
matcher = Matcher(nlp.vocab)

# Add the pattern
pattern = [{"Text": "iPhone"}, {"Text": "X"}]

# purpose of the parameters: Unique ID, optional callback, pattern
matcher.add("IPHONE_PATTERN", None, pattern)

doc = nlp("Upcoming iPhone X release date leaked in advance of iphone x launch")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
print("-------------------")

# Advanced pattern
pat = [{"IS_DIGIT": True}, {"LOWER": "fifa"}, {"LOWER": "world"}, 
       {"LOWER": "cup"}, {"IS_PUNCT": True}]

doc = nlp("2018 FIFA World Cup: France has won the fifa world cup in 2018")

matcher.add("FIFA", None, pat)
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
print("-------------------")

# Advanced pattern 
"""
match a generic form of buy (no part of speech specified), then match either
 0 or 1 determinates (specified by the OP:?), then match a noun
 
 OP can use one of 4 values:
     ! : negation, match no times
     ? : Optional, match 0 or 1 times
     + : Match 1 or more times
     * : Match 0 or more times
"""
pat = [{"LEMMA": "buy"}, {"POS": "DET", "OP": "?"}, {"POS": "NOUN"}]

doc = nlp("I bought a phone, now i must buy apps.  Not buying any headphones")

matcher.add("Phone", None, pat)
matches = matcher(doc)
for token in doc:
    print(token.text, token.pos_)
print("-------------------")
  
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)

# 12) Advanced Pattern Matching (on site code)
# Write one pattern that only matches mentions of the full iOS versions:
    # “iOS 7”, “iOS 11” and “iOS 10”.
print("\n-------------------")
    
doc = nlp(
    "After making the iOS update you won't notice a radical system-wide "
    "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
    "iOS 11's furniture remains the same as in iOS 10. But you will discover "
    "some tweaks once you delve a little deeper."
)

# Write a pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")
pattern = [{"TEXT": "iOS"}, {"IS_DIGIT": True}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("IOS_VERSION_PATTERN", None, pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)
    
print("-------------------")
    
# Write one pattern that only matches forms of “download” (tokens with the 
    # lemma “download”), followed by a token with the part-of-speech tag "PROPN" 
    # (proper noun).
doc = nlp(
    "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)

# Write a pattern that matches a form of "download" plus proper noun
pattern = [{"LEMMA": "download"}, {"POS": "PROPN"}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("DOWNLOAD_THINGS_PATTERN", None, pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)
