import pickle
import csv
import re
from time import time
from collections import defaultdict
from REALEC_extractor import RealecExtractor
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.util import align_tokens
from nltk.data import load

            

errors_to_correct = [
    # ('Prepositions',('Spelling',),PrepositionCorrector(),prepositions,RegexpParser('NP: {<IN>?<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*|PRP>}')),
    ('Articles',('Spelling','Prepositions','Prepositional_verb','Prepositional_adjective','Prepositional_adv','Prepositional_noun'))
]

tokenizer = TreebankWordTokenizer()
sent_tokenizer = load('tokenizers/punkt/{0}.pickle'.format('english'))

def tokenize(text):
    sents = sent_tokenizer.tokenize(text)
    sent_spans = sent_tokenizer.span_tokenize(text)
    tokens = [tokenizer.tokenize(sent) for sent in sents]
    idxs = [align_tokens([x.replace('``','"').replace("''",'"') 
                          if '``' in x or "''" in x else x for x in toks],
                         sent.replace('``','"').replace("''",'"'))
            for sent,toks in zip(sents,tokens)]
    return sents,tokens,idxs,sent_spans


for err,preverr in errors_to_correct:
    predsp = None
    predst = None
    correct = []
    all_sents = []
    tagged_sents = []
    init_sents = []

    tn = 0
    
    r = RealecExtractor(err,preverr,path_to_corpus='../REALEC')
    for text,error_spans in r.text_generator():
        if not tn % 100:
            print(tn)
        raw_sents,sents,idxs,sent_spans = tokenize(text)
        new_text = '\n'.join([' '.join(sent) for sent in sents])
        if new_text.strip()[-1] not in '.?!':
            new_text += '.'
        init_sents.append(new_text)
        tn += 1
    with open('init_sents_for_'+err.lower()+'.txt','w',encoding='utf-8') as f:
        f.write('\n==========.\n'.join(init_sents))


