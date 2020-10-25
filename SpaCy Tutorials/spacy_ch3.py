# -*- coding: utf-8 -*-`

import en_core_web_sm, en_core_web_md
# Chapter 3 Processing Pipelines

# 3) Inspecting the Pipeline Components
# Load the en_core_web_sm model
# nlpS = en_core_web_sm.load()
# nlpM = en_core_web_md.load()
# Print the names of the pipeline components
# print(nlpS.pipe_names)
# print(nlpM.pipe_names)

# # Print the full pipeline of (name, component) tuples
# print(nlpS.pipeline)
# print(nlpM.pipeline)

# 4) Custom Pipeline Components
"""
You can define your own pipeline function, and insert it anywhere in the 
order, so that you can have more control, or add different metadata to a doc
tag.  Example: Adding House names to the doc tokens consisting of known HP
chars with houses.  Harry -> Gryf, Cho -> Ravenclaw and such
"""

# def custom_component (doc):
#     # Do something to the doc
#     return doc

# nlpS.add_pipe(custom_component)
"""add_pipe has 4 additional optional arguments:
    last:    If true, adds new compnent last
    first:   If true, adds new component first
    before:  Add before specific already existing comp
    after:   Add after already existing component
"""  

# 6) Simple Custom Components
# create the nlp object
nlp = en_core_web_sm.load()

# Define the custom component
def length_component (doc):
    # Print doc length and return the doc object
    doc_length = len(doc)
    print(f"This document is {doc_length} tokens long.")
    return doc

# Have it load in the pipeline first
nlp.add_pipe(length_component, first = True)

print("Pipeline: ", nlp.pipe_names)
# Pipeline:  ['custom_component', 'tagger', 'parser', 'ner']

doc = nlp("Hi there world!\n----\n")
print("\n----\n")
# Now will automaticlaly print out the number of tokens in the doc

# 7) Complex Components
"""
In this exercise, you’ll be writing a custom component that uses the 
PhraseMatcher to find animal names in the document and adds the matched 
spans to the doc.ents. A PhraseMatcher with the animal patterns has already
been created as the variable matcher.

Define the custom component and apply the matcher to the doc.
Create a Span for each match, assign the label ID for "ANIMAL" and overwrite
 the doc.ents with the new spans.
Add the new component to the pipeline after the "ner" component.
Process the text and print the entity text and entity label for the entities
 in doc.ents.
"""
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

animals = ["Golden Retriever", "cat", "turtle", "Rattus norvegicus"]
animal_patterns = list(nlp.pipe(animals))
print("animal_patterns:", animal_patterns)
matcher = PhraseMatcher(nlp.vocab)
matcher.add("ANIMAL", None, *animal_patterns)

# Define the custom component
def animal_component(doc):
    # Apply the matcher to the doc
    matches = matcher(doc)
    # Create a Span for each match and assign the label "ANIMAL"
    spans = [Span(doc, start, end, label="ANIMAL")
             for match_id, start, end in matches]
    # Overwrite the doc.ents with the matched spans
    doc.ents = spans
    return doc


# Add the component to the pipeline after the "ner" component
nlp.add_pipe(animal_component, after = "ner")
print(nlp.pipe_names)

# Process the text and print the text and label for the doc.ents
doc = nlp("I have a cat and a Golden Retriever")
print([(ent.text, ent.label) for ent in doc.ents])
print("\n----\n")





















