from nltk import StanfordPOSTagger
from nltk import RegexpParser
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
from numpy import vstack
from gensim.models import KeyedVectors

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from feature_extraction.article_extraction import create_article_rows
from postprocessing.articles import ArticleCorrector
from postprocessing.prepositions import PrepositionCorrector
from preprocessing.conllu import parse_tree


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

with open('../prepositions.txt','r',encoding='utf-8') as f:
    prepositions = f.read().split('\n')
    prepositions.append('zero')

def change_conllu(conllu_str):
    spl_pattern = '\n# sent_id = [0-9]+\n# text = ==========\n1\t==========\t==========\tSYM\t\$\t_\t0\troot\t_\tSpacesAfter=.+\|TokenRange=[0-9]+\:[0-9]+\n'
    texts = re.split(spl_pattern,conllu_str)
    new_texts = []
    for text in texts:
        lines = text.split('\n')
        first_idx = None
        new_lines = []
        for line in lines:
            if line.startswith('#') or not line.strip():
                new_lines.append(line)
                continue
            line_elements = line.strip().split('\t')
            token_range = line_elements[-1].split('|')[-1].split('=')[-1]
            #print(line,line_elements,token_range)
            if first_idx is None:
                first_idx = int(token_range.split(':')[0])
            new_token_range = ':'.join([str(int(x)-first_idx) for x in token_range.split(':')])
            new_line = line.strip()[:-len(token_range)] + new_token_range
            #print(token_range,new_token_range,first_idx,new_line)
            new_lines.append(new_line)
        new_texts.append('\n'.join(new_lines).strip())
    return new_texts
            
            

errors_to_correct = [
    ('Prepositions',('Spelling',),PrepositionCorrector(),prepositions,RegexpParser('NP: {<IN>?<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*|PRP>}')),
    #('Articles',('Spelling',),ArticleCorrector(),['a','an','the','zero'],RegexpParser(r'NP: {<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*>}'))
]

w2v_model = KeyedVectors.load_word2vec_format(
            "C:/Users/PC1/Desktop/python/деплом/deplom/constructions/GoogleNews-vectors-negative300.bin.gz",
            binary=True)
#w2v_model = None

#regexp-based chunker

for err,preverr,corrector,options,chunker in errors_to_correct:
    predsp = None
    predst = None
    correct = []
    all_sents = []
    tagged_sents = []
    init_sents = []

    tn = 0
    with open('tagged_sents_for_'+err.lower()+'.pickle','rb') as f:
        tagged_sents = pickle.load(f)
        
    with open('init_sents_for_'+err.lower()+'_parsed.txt','r',encoding='utf-8') as f:
        new_parsed = change_conllu(f.read())
        print(len(new_parsed))
        trees = [parse_tree(x) for x in new_parsed]
    
    r = RealecExtractor(err,preverr,path_to_corpus='../REALEC/exam/exam2014')
    for text,error_spans in r.text_generator():
        #if tn > 5:
        #    break
        #print(tn,text,error_spans)
        if not tn % 100:
            print(tn)
        raw_sents,sents,idxs,sent_spans = tokenize(text)
        #tsents = st.tag_sents(sents)
        #tagged_sents.append(tsents)
        #if text.strip()[-1] not in '.?!':
        #    text += '.'
        #init_sents.append(text)
        tsents = tagged_sents[tn]
        curr_trees = trees[tn]
        corrector.get_table(sents,tsents,idxs,raw_sents,sent_spans,curr_trees)
        #corrector.feats.to_csv('first_feats.csv',sep=';',encoding='utf-8-sig')
        corrector.get_feature_matrix(w2v_model)
        predsp_curr, predst_curr = corrector.get_probas()
        corrector.feats['Predicted'] = corrector.get_first_preds()
        predsp = vstack((predsp,predsp_curr)) if predsp is not None else predsp_curr
        predst = vstack((predst,predst_curr)) if predst is not None else predst_curr
        i = 0
        for np in corrector.feats.itertuples():
            #print(np)
            i, curr_correct = corrector.get_error_span(np,i,error_spans)
            correct.append(curr_correct)
            curr_sents = []
            for option in options:
                #print(np)
                new_pred = corrector.form_one_correction(np.raw_NP, option, np.Start_idx)
                corr_sent = correct_text(np.Sentence,[(np.Start_idx,np.raw_NP,new_pred)])
                curr_sents.append(' '.join(tokenizer.tokenize(corr_sent)))
            all_sents.append(curr_sents)
                
        #if i != len(error_spans):
        #    print(text)
        #    print(i)
        #    print(error_spans)
        #    print(article_corrector.feats[['raw_NP','Start_idx','Sent_start_idx']])
        #    print('=================')
        corrector.feats = []
        tn += 1
    #with open('init_sents_for_'+err.lower()+'.txt','w',encoding='utf-8') as f:
    #    f.write('\n==========\n\n'.join(init_sents))
        
    #with open('tagged_sents_for_'+err.lower()+'.pickle','wb') as f:
    #    pickle.dump(tagged_sents,f)


    with open(err.lower()+'_meta.csv','w',encoding='utf-8-sig',newline='') as f:
        csvw = csv.writer(f,delimiter=';',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csvw.writerow(corrector.logit_bin.classes_.tolist() + corrector.logit_type.classes_.tolist() +
                     ['raw_NP','Start_idx','Sent_start_idx','Initial','ML_L1','Ann'])
        for pred,predt,corr in zip(predsp,predst,correct):
            csvw.writerow(list(pred)+list(predt)+corr)

    with open('sents_'+err.lower()+'.txt','w',encoding='utf-8') as f:
        f.write('\n\n'.join(['\n'.join(x) for x in all_sents]))
        f.write('\n')



