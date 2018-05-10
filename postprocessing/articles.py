import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix, lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TreebankWordTokenizer
from nltk import RegexpParser
from collections import defaultdict
import re
import json


import sys, os
sys.path.insert(0, os.path.abspath('..'))
from feature_extraction.article_extraction import create_article_rows
from feature_extraction.lm_probas import get_lm_probas


class ArticleCorrector:
    def __init__(self):
        grammar = r'NP: {<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*>}'
        self.chunker = RegexpParser(grammar)
        self.tokenizer = TreebankWordTokenizer()

        self.cuvplus = defaultdict(list)
        with open('../cuvplus.txt','r',encoding='utf-8-sig') as f:
            for line in f.readlines():
                entry = line.strip().split('|')
                self.cuvplus[entry[0]].append(entry[1:])

        self.feats = []
        
        self.cols = ['Sentence','raw_NP','NP','Start_idx','Sent_start_idx','POS_tags','Head',
                       'Head_countability','NP_first_letter','Head_POS',
                       'hypernyms','higher_hypernyms',#'hhead','hhead_POS','deprel',
                       'prevprev','prev','post','postpost','prevprev_POS','prev_POS',
                       'post_POS','postpost_POS','Article']

        with open('../models/article_logit_binary.pickle','rb') as f:
           self.logit_bin = pickle.load(f)
            
        with open('../models/article_logit_type.pickle','rb') as f:
           self.logit_type = pickle.load(f)

        with open('../models/article_metaclassifier_forest.pickle','rb') as f:
            self.metaforest = pickle.load(f)

        self.subst = re.compile('^[aA][nN]? |^[tT][hH][eE] ')


    def get_table(self,sents,tsents,idxs,raw_sents,sent_spans):
        for sent,tsent,spans,raw_sent,sent_span in zip(sents,tsents,idxs,raw_sents,sent_spans):
            if ' '.join(sent) == ' '.join(sent).upper():
                sent = [x.lower() for x in sent if x]
            else:
                sent = [x for x in sent if x]
            self.feats.extend(create_article_rows(sent,tsent,self.chunker,self.cuvplus,
                                                  spans,raw_sent,sent_span[0]))
        self.feats = pd.DataFrame(self.feats,columns=self.cols)
        

    def get_feature_matrix(self,w2v_model):
        with open('../models/one_word_vectorizer.pickle','rb') as f:
           onewordvect = pickle.load(f)

        with open('../models/pos_vectorizer.pickle','rb') as f:
            pos_vect = pickle.load(f)

        with open('../models/letter_vectorizer.pickle','rb') as f:
           letter_vect = pickle.load(f)

        with open('../models/np_vectorizer.pickle','rb') as f:
           np_vect = pickle.load(f)
            
        with open('../models/noun_hypernym_vectorizer.pickle','rb') as f:
           hyp_vect = pickle.load(f)
            
        with open('../models/noun_higher_hypernym_vectorizer.pickle','rb') as f:
           hhyp_vect = pickle.load(f)

        with open('../models/countability_vectorizer.pickle','rb') as f:
           count_vect = pickle.load(f)

        all_vectors = lil_matrix((self.feats.shape[0],300))
        #for i,word in enumerate(self.feats['Head']):
        #    if word in w2v_model:
        #        all_vectors[i,:] = w2v_model[word]

        npm = np_vect.transform(self.feats['NP'])
        pos = pos_vect.transform(self.feats['POS_tags'])
        head_pos = pos_vect.transform(self.feats['Head_POS'])
        prevprev_pos = pos_vect.transform(self.feats['prevprev_POS'])
        prev_pos = pos_vect.transform(self.feats['prev_POS'])
        post_pos = pos_vect.transform(self.feats['post_POS'])
        postpost_pos = pos_vect.transform(self.feats['postpost_POS'])
        countability = count_vect.transform(self.feats['Head_countability'])
        first_letter = letter_vect.transform(self.feats['NP_first_letter'])
        hyp = hyp_vect.transform(self.feats['hypernyms'])
        hhyp = hhyp_vect.transform(self.feats['higher_hypernyms'])
        head = onewordvect.transform(self.feats['Head'])
        prevprev = onewordvect.transform(self.feats['prevprev'])
        prev = onewordvect.transform(self.feats['prev'])
        post = onewordvect.transform(self.feats['post'])
        postpost = onewordvect.transform(self.feats['postpost'])

        self.data_sparse = hstack((npm,pos,head,countability,first_letter,head_pos,
                              hyp,hhyp,all_vectors,prevprev,prev,post,postpost,
                              prevprev_pos,prev_pos,post_pos,postpost_pos)).tocsr()


    def get_probas(self):
        preds = self.logit_bin.predict_proba(self.data_sparse)
        preds_type = self.logit_type.predict_proba(self.data_sparse)
        return preds, preds_type


    def get_first_preds(self):
        preds = self.logit_bin.predict(self.data_sparse)
        preds_type = self.logit_type.predict(self.data_sparse[preds == 'present'])
        preds[preds == 'present'] = preds_type
        self.feats['Predicted'] = preds
        return preds

    def process_probs(self,probs):
        init_prob = []
        corr_prob = []
        for i in range(0,len(probs),2):
            init_prob.append(probs[i])
            if probs[i+1] is None:
                corr_prob.append(probs[i])
            else:
                corr_prob.append(probs[i+1])
        return init_prob,corr_prob


    def get_sentences_for_lm(self):
        sents = []
        for i,np,sent,start_idx,sent_idx,article,pred in \
            self.feats[['raw_NP','Sentence','Start_idx','Sent_start_idx','Article','Predicted']].itertuples():
            if article == pred:
                sents.extend((' '.join(self.idx_to_sent[sent_idx]),''))
            else:
                new_pred = self.form_one_correction(np,pred)
                new_sent = sent[:start_idx]+new_pred+sent[start_idx+len(np):]
                sents.extend((' '.join(self.idx_to_sent[sent_idx]),' '.join(self.tokenizer.tokenize(new_sent))))
        return '\n'.join(sents)+'\n'
            

    def write_sentences_for_lm(self,sents):
        with open('test_sents.txt','w',encoding='utf-8') as f:
            f.write(sents)


    def get_metamatrix(self):
        preds, preds_type = self.get_probas()
        str_sents = self.get_sentences_for_lm()
        # probs = get_lm_probas(str_sents)
        # temp for local windows - file is processed by lm on server separately
        #self.write_sentences_for_lm(str_sents)
        with open('lm_preds_test.json','r',encoding='utf-8') as f:
            probs = json.loads(f.read())
        # end temp
        self.metafeats = pd.concat([pd.DataFrame(preds),pd.DataFrame(preds_type)],axis=1)
        self.metafeats['init_prob'],self.metafeats['corr_prob'] = self.process_probs(probs)
        self.metafeats['probs_ratio'] = self.metafeats['init_prob'] / self.metafeats['corr_prob']
        self.metafeats['probs_delta'] = self.metafeats['init_prob'] - self.metafeats['corr_prob']
        for sent,np,iprob,cprob in zip(self.feats['Sentence'],self.feats['raw_NP'],
                                       self.metafeats['init_prob'],self.metafeats['corr_prob']):
            print(sent,np,iprob,cprob)

        with open('../models/article_choice_vectorizer.pickle','rb') as f:
            art_vect = pickle.load(f)
        self.metafeats = hstack((self.metafeats.to_sparse(),art_vect.transform(self.feats['Article']),
                                 art_vect.transform(self.feats['Predicted'])))
        print(self.metafeats.shape)


    def get_preds(self):
        preds = self.metaforest.predict(self.metafeats)
        return preds
        
    def form_one_correction(self,initial,predicted):
        repl = predicted+' ' if predicted != 'zero' else ''
        if any([initial.lower().startswith(x) for x in ['a ','an ','the ']]):
            predicted = self.subst.sub(repl,initial)
        else:
            predicted = repl + initial
        return predicted

    def form_corrections(self,preds):
        to_correct = self.feats.loc[self.feats['Predicted'] != self.feats['Article'],:]
        to_correct = to_correct[['Start_idx','Sent_start_idx','raw_NP','Predicted']].values.tolist()
        #print(to_correct)
        for i in range(len(to_correct)):
            to_correct[i][3] = self.form_one_correction(to_correct[i][2],to_correct[i][3])
        return to_correct

    def detect_errors(self,w2v_model,sents,tsents,idxs,raw_sents,sent_spans):
        self.sents = sents
        self.tsents = tsents
        self.idx_to_sent = {idx[0]:sent for idx,sent in zip(sent_spans,sents)}
        print('\tGetting table')
        self.get_table(sents,tsents,idxs,raw_sents,sent_spans)
        print('\tGetting feature matrix')
        self.get_feature_matrix(w2v_model)
        print('\tGetting first preds')
        self.get_first_preds()
        print('\tGetting table for metaclassifier')
        self.get_metamatrix()
        print('\tGetting final preds')
        preds = self.get_preds()
        print('\tForming corrections')
        corrs = self.form_corrections(preds)
        return corrs
        



# сгенерить табличку на основе текста (включая tag_sents и чанкинг)
# предсказать по табличке
# тут должна быть вся артиклевая система, но пока только L1
# выдача - стартовый индекс в старом тексте, старая NP, новая NP
