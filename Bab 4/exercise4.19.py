# Example 4.23
# Sentiment Analysis using Hugging Face Transformers

from transformers import pipeline

# Create sentiment analysis classifier
classifier = pipeline('sentiment-analysis')


# Different sentences
sentences = [
    "This movie is absolutely amazing and very entertaining.",
    "I really enjoyed the story and the acting was excellent.",
    "The food was delicious and the service was great.",
    "This movie was boring and a waste of time.",
    "I did not like the story and the acting was terrible.",
    "The service was slow and the food was disappointing."
]


# Analyze each sentence
for sentence in sentences:

    result = classifier(sentence)

    print("Sentence :", sentence)
    print("Result   :", result)
    print()