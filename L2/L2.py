from nltk import StanfordPOSTagger
from nltk import RegexpParser
import pickle
import csv
from time import time
from collections import defaultdict
from REALEC_extractor import RealecExtractor
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.util import align_tokens
from nltk.data import load
from numpy import vstack
from gensim.models import KeyedVectors

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from feature_extraction.article_extraction import create_article_rows
from postprocessing.articles import ArticleCorrector


st = StanfordPOSTagger('../stanfordPOStagger/english-bidirectional-distsim.tagger',
                       '../stanfordPOStagger/stanford-postagger.jar',
                       java_options='-mx2048m')

tokenizer = TreebankWordTokenizer()
sent_tokenizer = load('tokenizers/punkt/{0}.pickle'.format('english'))

def tokenize(text):
    sents = sent_tokenizer.tokenize(text)
    sent_spans = sent_tokenizer.span_tokenize(text)
    tokens = [tokenizer.tokenize(sent) for sent in sents]
    idxs = [align_tokens(['"' if x in ['``',"''"] else x for x in toks],sent)
            for sent,toks in zip(sents,tokens)]
    return sents,tokens,idxs,sent_spans


def normalize_article_corr(corr):
    if corr == 'DELETE':
        return 'zero'
    toks = corr.split()
    if toks[0].lower() in {'a','an','the'}:
        return toks[0].lower()
    else:
        return 'zero'

def correct_text(text,corrs):
    # given corrections (start idx,initial text, correction), correct the text
    slices = []
    last_idx = 0
    for item in corrs:
        idx,initial,corr = item
        slices.append(text[last_idx:idx])
        slices.append(corr)
        last_idx = idx + len(initial)
    slices.append(text[last_idx:])
    return ''.join(slices)


errors_to_correct = [
    ('Articles',('Spelling',),normalize_article_corr)
]

article_corrector = ArticleCorrector()
w2v_model = KeyedVectors.load_word2vec_format(
            "C:/Users/PC1/Desktop/python/деплом/deplom/constructions/GoogleNews-vectors-negative300.bin.gz",
            binary=True)
#w2v_model = None

#regexp-based chunker
grammar = r'NP: {<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*>}'
chunker = RegexpParser(grammar)

predsp = None
predst = None
correct = []
all_sents = []
tagged_sents = []
tn = 0
for err, preverr,norm_func in errors_to_correct:
    r = RealecExtractor(err,preverr,path_to_corpus='../REALEC/exam/exam2014')
    for text,error_spans in r.text_generator():
        #if tn > 3:
        #    break
        #print(text,error_spans)
        #if not tn % 100:
        #    print(tn)
        print(tn)
        raw_sents,sents,idxs,sent_spans = tokenize(text)
        idx_to_sent = {idx[0]:sent for idx,sent in zip(sent_spans,sents)}
        tsents = st.tag_sents(sents)
        tagged_sents.append(tsents)
        article_corrector.get_table(sents,tsents,idxs,raw_sents,sent_spans)
        article_corrector.get_feature_matrix(w2v_model)
        predsp_curr, predst_curr = article_corrector.get_probas()
        article_corrector.feats['Predicted'] = article_corrector.get_preds()
        predsp = vstack((predsp,predsp_curr)) if predsp is not None else predsp_curr
        predst = vstack((predst,predst_curr)) if predst is not None else predst_curr
        i = 0
        for np in article_corrector.feats.itertuples():
            #print(np)
            while i < len(error_spans) and \
                  error_spans[i][0] < np.Sent_start_idx + np.Start_idx - 1:
                i += 1
            if i < len(error_spans) and \
               error_spans[i][0] > np.Sent_start_idx + np.Start_idx - 2 and \
               error_spans[i][1] < np.Sent_start_idx + len(np.raw_NP) + np.Start_idx + 2:
                correct.append([np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Article,np.Predicted,norm_func(error_spans[i][2])])
                i += 1
            else:
                correct.append([np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Article,np.Predicted,np.Article])
            if np.Article.lower() == np.Predicted:
                all_sents.extend([' '.join(idx_to_sent[np.Sent_start_idx]),''])
            else:
                #print(np)
                new_pred = article_corrector.form_one_correction(np.raw_NP, np.Predicted)
                corr_sent = correct_text(np.Sentence,[(np.Start_idx,np.raw_NP,new_pred)])
                all_sents.extend([' '.join(idx_to_sent[np.Sent_start_idx]),
                                  ' '.join(tokenizer.tokenize(corr_sent))])
                
        #if i != len(error_spans):
        #    print(text)
        #    print(i)
        #    print(error_spans)
        #    print(article_corrector.feats[['raw_NP','Start_idx','Sent_start_idx']])
        #    print('=================')
        article_corrector.feats = []
        tn += 1


with open('articles_meta.csv','w',encoding='utf-8-sig',newline='') as f:
    csvw = csv.writer(f,delimiter=';',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    for pred,predt,corr in zip(predsp,predst,correct):
        csvw.writerow(list(pred)+list(predt)+corr)

with open('sents.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(all_sents))
    f.write('\n')

with open('tagged_sents_2014_for_articles.pickle','wb') as f:
    pickle.dump(tagged_sents,f)


