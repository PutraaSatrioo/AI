# Example 4.22 - NLTK
# Text Analysis with Different Text

import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')


# =========================================================
# Different text for analysis
# =========================================================

sentence = """
Microsoft was founded by Bill Gates and Paul Allen in 1975.
The company developed Windows and Microsoft Office, which became
widely used around the world. Today, Microsoft is headquartered
in Redmond, Washington, and develops technologies related to
cloud computing, artificial intelligence, and software.
"""


# =========================================================
# Tokenization
# =========================================================

tokens = nltk.word_tokenize(sentence)

print("TOKENS:")
print(tokens)


# =========================================================
# Part-of-Speech Tagging
# =========================================================

tagged = nltk.pos_tag(tokens)

print("\nPOS TAGGING:")
print(tagged)


# =========================================================
# Named Entity Recognition
# =========================================================

entities = nltk.chunk.ne_chunk(tagged)

print("\nNAMED ENTITIES:")
print(entities)