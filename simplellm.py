
import os
from dotenv import load_dotenv
load_dotenv()

import openai
openai.api_key=os.getenv("OPENAI_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")
print(groq_api_key)
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
print(model)

from langchain_core.messages import HumanMessage,SystemMessage

messages=[
    SystemMessage(content="Translate the following from English to French"),
    HumanMessage(content="Hello How are you?")
]

result=model.invoke(messages)

print(result)

from langchain_core.output_parsers import StrOutputParser
parser=StrOutputParser()
print(parser.invoke(result))

### Using LCEL- chain the components
chain=model|parser
print(chain.invoke(messages))
### Prompt Templates
from langchain_core.prompts import ChatPromptTemplate

generic_template="Trnaslate the following into {language}:"

prompt=ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")]
)

result=prompt.invoke({"language":"French","text":"Hello"})

print(result.to_messages())

##Chaining together components with LCEL
chain=prompt|model|parser
print(chain.invoke({"language":"French","text":"Hello"}))
