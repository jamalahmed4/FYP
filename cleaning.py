import os 
import spacy 
import string 
import re 
from tqdm import tqdm

os.system('pip install -q spacy')

os.system('! spacy download en_core_web_sm')
os.system('! spacy download zh_core_web_sm')
os.system('! spacy download nl_core_news_sm')
os.system('! spacy download fr_core_news_sm')
os.system('! spacy download de_core_news_sm')
os.system('! spacy download ja_core_news_sm')
os.system('! spacy download pl_core_news_sm')
os.system('! spacy download ru_core_news_sm')
os.system('! spacy download es_core_news_sm')

letters = list(string.ascii_letters)
digits = list(string.digits)
punctuations = ['.' , ',' , ' ']
usefull_characs = ''.join(list(letters + digits + punctuations))
usefull_pattern = f"[^{re.escape(usefull_characs)}]"
link_pattern = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"


ner_model_wrapper = {
    'en' : spacy.load('en_core_web_sm') , # English
    'zh' : spacy.load('zh_core_web_sm') , # Chinese
    'nl' : spacy.load('nl_core_news_sm') , # Dutch
    'fr' : spacy.load('fr_core_news_sm') , # French
    'de' : spacy.load('de_core_news_sm') , # German
    'ja' : spacy.load('ja_core_news_sm') , # Japenese
    'pl' : spacy.load('pl_core_news_sm') , # Polish
    'ru' : spacy.load('ru_core_news_sm') , # Russian
    'es' : spacy.load('es_core_news_sm') # Spanish
}

def clean_text(text , languages = ['en'] , ner_chunk_length = 50000) : 

    ner_models = [
        ner_model_wrapper[language]
        for language
        in languages
    ]

    chunks = [
        text[index : index + ner_chunk_length] # Limit is around 49K bytes
        for index
        in range(0 , len(text) , ner_chunk_length)
    ]

    entities = []

    for ner_model in tqdm(ner_models , total = len(ner_models) , desc = 'Detecting NER for Different Languages ===>') : # Takes a little bit of time and RAM for this around 200MB and around 20 seconds

        for chunk in tqdm(chunks , total = len(chunks) , desc = f'Detecting NER for {ner_model} ===>') :

            ents = ner_model(chunk).ents

            for ent in ents :

                if ent.label_ == 'PERSON' : entities.append(str(ent))

    for entity in entities : text = text.replace(entity , '')

    text = re.sub(link_pattern, "", text)
    text = re.sub(usefull_pattern, "", text)

    return text