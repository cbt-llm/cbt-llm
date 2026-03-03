# deepseek-r1:14b

from ollama import chat

INPUT=""

response = chat(
    model='deepseek-r1:14b',
    messages=INPUT,
    response_format={
        'type': 'json_object'
    }
)
print(response.message.content)