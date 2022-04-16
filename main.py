import numpy as np # Math
import requests # Getting text from websites
import html2text # Converting wiki pages to plain text
from googlesearch import search # Performing Google searches
import re
from transformers import pipeline
from bs4 import BeautifulSoup
from markdown import markdown
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

model_name = 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'
nlp = pipeline("question-answering", model=model_name)

# Source: https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe
def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text

def format_text(text):
    text = markdown_to_text(text)
    text = text.replace('\n', ' ')
    return text

def query_pages(query, n=2):
    return list(search(query, stop=n))

def query_to_text(query, n=2):
    html_conv = html2text.HTML2Text()
    html_conv.ignore_links = True
    html_conv.escape_all = True
    
    text = []
    for link in query_pages(query, n):
        req = requests.get(link)
        text.append(html_conv.handle(req.text))
        text[-1] = format_text(text[-1])
        
    return text

def link_to_text(lista):
    html_conv = html2text.HTML2Text()
    html_conv.ignore_links = True
    html_conv.escape_all = True
    
    text = []
    for link in lista:
        req = requests.get(link)
        text.append(html_conv.handle(req.text))
        text[-1] = format_text(text[-1])
        
    return text

def join_text(context):
  tt = ''

  for pos in context:
      tt += pos + ' '

  tt = tt.rstrip()

  return tt

def q_to_a(question, n=2, debug=False):
    context = query_to_text(question, n=n)
    context = join_text(context)
    pred = nlp(question=question, context=context)
    return pred['answer']

@app.get("/items/{person_id}")
def read_item(person_id: int, question: str, pages: Optional[int] = 2, context: Optional[str] = None):
  if not context or context == "":
    context = query_to_text(question, n=pages)
    context = join_text(context)

  elif "https://" in context or "http://" in context:
    context = link_to_text([context])
    context = join_text(context)

  result = nlp(question=question, context=context)["answer"]

  return {"reponse": result}