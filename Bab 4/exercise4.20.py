# Example 4.24
# Question Answering using Hugging Face Transformers

from transformers import pipeline

# Create question answering pipeline
question_answerer = pipeline('question-answering')


# Different context
context = """
PENS was founded in 1988 in Surabaya, Indonesia.
The institution focuses on electronics, information technology,
and engineering education. PENS has several study programs,
including Multimedia Engineering Technology. The campus is
located in Sukolilo, Surabaya.
"""


# Different questions
questions = [
    "When was PENS founded?",
    "Where is PENS located?",
    "What fields does PENS focus on?",
    "Which study program is mentioned in the text?",
    "In which city is the campus located?"
]


# Ask each question
for question in questions:

    result = question_answerer({
        'question': question,
        'context': context
    })

    print("Question :", question)
    print("Answer   :", result['answer'])
    print("Score    :", round(result['score'] * 100, 2), "%")
    print()