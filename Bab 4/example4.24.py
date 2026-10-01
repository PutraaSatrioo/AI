#Example 4.24
from transformers import pipeline

question_answer = pipeline('question-answering')

question_answer({
    'question': 'What is the name of the company?',
    'context': 'We created Biox Systems Ltd company back in the year of 2000.'
})