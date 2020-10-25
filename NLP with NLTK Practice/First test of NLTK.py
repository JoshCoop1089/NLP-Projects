# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:54:12 2020

@author: joshc

Before attempting this, had to do 

import nltk
nltk.download

to dl and import the packages, methods, and corpus for messing around.
I think this is a one time thing 

"""

from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)
