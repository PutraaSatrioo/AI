# Example 4.23
# https://github.com/huggingface/transformers
# pip install transformers

from transformers import pipeline

classifier = pipeline('sentiment-analysis')

print(classifier('This is a good movie.'))