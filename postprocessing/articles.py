import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix, lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
from nltk.tokenize import TreebankWordTokenizer
from nltk import RegexpParser
from collections import defaultdict
import re
import json
from article_extraction import create_article_rows
from lm_probas import get_lm_probas


class ArticleCorrector:
    def __init__(self):
        grammar = r'NP: {<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*>}'
        self.chunker = RegexpParser(grammar)
        self.tokenizer = TreebankWordTokenizer()
        self.options = ['a','an','the','zero']

        self.cuvplus = defaultdict(list)
        with open('../cuvplus.txt','r',encoding='utf-8-sig') as f:
            for line in f.readlines():
                entry = line.strip().split('|')
                self.cuvplus[entry[0]].append(entry[1:])

        self.feats = []
        
        self.cols = ['Sentence','raw_NP','NP','Start_idx','Sent_start_idx','POS_tags','Head',
                       'Head_countability','NP_first_letter','Head_POS',
                       'hypernyms','higher_hypernyms',#'hhead','hhead_POS','deprel',
                       'prev_2','prev_1','prev_2_POS','prev_1_POS',
                       'post_1','post_2','post_1_POS','post_2_POS',
                       'Target']

        with open('../models/article_logit_binary.pickle','rb') as f:
           self.logit_bin = pickle.load(f)
            
        with open('../models/article_logit_type.pickle','rb') as f:
           self.logit_type = pickle.load(f)

        self.subst = re.compile('^[aA][nN]? |^[tT][hH][eE] ')


    def get_table(self,sents,tsents,idxs,raw_sents,sent_spans):
        for sent,tsent,spans,raw_sent,sent_span in zip(sents,tsents,idxs,raw_sents,sent_spans):
            if ' '.join(sent) == ' '.join(sent).upper():
                sent = [x.lower() for x in sent if x]
            else:
                sent = [x for x in sent if x]
            self.feats.extend(create_article_rows(sent,tsent,self.chunker,self.cuvplus,
                                                  None,spans,raw_sent,sent_span[0]))
        self.feats = pd.DataFrame(self.feats,columns=self.cols)
        

    def get_feature_matrix(self):
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

        npm = np_vect.transform(self.feats['NP'])
        pos = pos_vect.transform(self.feats['POS_tags'])
        head_pos = pos_vect.transform(self.feats['Head_POS'])
        prevs_pos = hstack([pos_vect.transform(self.feats['prev_'+str(i)+'_POS']) for i in range(1,3)])
        posts_pos = hstack([pos_vect.transform(self.feats['post_'+str(i)+'_POS']) for i in range(1,3)])
        countability = count_vect.transform(self.feats['Head_countability'])
        first_letter = letter_vect.transform(self.feats['NP_first_letter'])
        hyp = hyp_vect.transform(self.feats['hypernyms'])
        hhyp = hhyp_vect.transform(self.feats['higher_hypernyms'])
        head = onewordvect.transform(self.feats['Head'])
        prevs = hstack([onewordvect.transform(self.feats['prev_'+str(i)]) for i in range(1,3)])
        posts = hstack([onewordvect.transform(self.feats['post_'+str(i)]) for i in range(1,3)])

        self.data_sparse = hstack((npm,pos,head,countability,first_letter,head_pos,
                              hyp,hhyp,prevs,posts,prevs_pos,posts_pos)).tocsr()

        with open('../models/nonzero_columns.pickle','rb') as f:
            nonzero_columns = pickle.load(f)
        self.data_sparse = self.data_sparse[:,nonzero_columns]


    def get_probas(self):
        preds = self.logit_bin.predict_proba(self.data_sparse)
        preds_type = self.logit_type.predict_proba(self.data_sparse)
        return preds, preds_type


    def get_first_preds(self):
        preds = self.logit_bin.predict(self.data_sparse)
        if not self.data_sparse[preds == 'present'].shape[0]:
            self.feats['Predicted'] = preds
            return preds
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

    def normalize_article_corr(self,corr):
        if corr == 'DELETE':
            return 'zero'
        toks = corr.split()
        if toks[0].lower() in {'a','an','the'}:
            return toks[0].lower()
        else:
            return 'zero'

    def get_error_span(self,np,i,error_spans):
        while i < len(error_spans) and \
              error_spans[i][0] < np.Sent_start_idx + np.Start_idx - 1:
            i += 1
        if i < len(error_spans) and \
           error_spans[i][0] > np.Sent_start_idx + np.Start_idx - 2 and \
           error_spans[i][1] < np.Sent_start_idx + len(np.raw_NP) + np.Start_idx + 2:
            return i+1, [np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Target,np.Predicted,
                         self.normalize_article_corr(error_spans[i][2])]
        else:
            return i,[np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Target,np.Predicted,np.Target]

    def get_sentences_for_lm(self):
        sents = []
        for i,np,sent,start_idx,sent_idx,article,pred in \
            self.feats[['raw_NP','Sentence','Start_idx','Sent_start_idx','Target','Predicted']].itertuples():
            curr_sents = []
            for option in self.options:
                new_pred = self.form_one_correction(np,option,start_idx)
                new_sent = sent[:start_idx]+new_pred+sent[start_idx+len(np):]
                curr_sents.append(' '.join(self.tokenizer.tokenize(new_sent)))
            sents.append(curr_sents)
        return sents
            

    def write_sentences_for_lm(self,sents):
        with open('test_sents_articles.txt','w',encoding='utf-8') as f:
            f.write('\n\n'.join(['\n'.join(x) for x in sents]))
            f.write('\n')


    def get_metamatrix(self):
        preds, preds_type = self.get_probas()
        str_sents = self.get_sentences_for_lm()
        probs = get_lm_probas(str_sents)
        # temp for local windows - file is processed by lm on server separately
        #self.write_sentences_for_lm(str_sents)
        #with open('lm_preds_test_articles.json','r',encoding='utf-8') as f:
        #    probs = json.loads(f.read())
        # end temp
        self.metafeats = pd.concat([pd.DataFrame(preds,columns=['present','zero']),
                                    pd.DataFrame(preds_type,columns=['a','an','the']),
                                    pd.DataFrame(probs,columns=['lm_a','lm_an','lm_the','lm_zero'])],
                                   axis=1)
        probs_ratio = []
        probs_delta = []
        init_probs = []
        corr_probs = []
        lm_choice = []
        for i in range(self.metafeats.shape[0]):
            row = self.metafeats.iloc[i]
            feat_row = self.feats.iloc[i]
            init_prob = row['lm_'+feat_row['Target']]
            corr_prob = row['lm_'+feat_row['Predicted']]
            init_probs.append(init_prob)
            corr_probs.append(corr_prob)
            probs_ratio.append(init_prob / corr_prob)
            probs_delta.append(init_prob - corr_prob)
            lm_choice.append(np.argmax(row[['lm_a','lm_an','lm_the','lm_zero']]).split('_')[1])
        self.metafeats['init_prob'] = init_probs
        self.metafeats['corr_prob'] = corr_probs
        self.metafeats['probs_ratio'] = probs_ratio
        self.metafeats['probs_delta'] = probs_delta
        self.feats['LM'] = lm_choice
        #for sent,np,iprob,cprob in zip(self.feats['Sentence'],self.feats['raw_NP'],
        #                               self.metafeats['init_prob'],self.metafeats['corr_prob']):
        #    print(sent,np,iprob,cprob)

        self.metafeats = self.metafeats.loc[(self.feats['Target'] != self.feats['Predicted']) |
                                            (self.feats['Target'] != self.feats['LM']),:]

        with open('../models/article_choice_vectorizer.pickle','rb') as f:
            art_vect = pickle.load(f)
        self.metafeats_sparse = hstack((self.metafeats.to_sparse(),
                                 art_vect.transform(self.feats.loc[self.metafeats.index,'Target']),
                                 art_vect.transform(self.feats.loc[self.metafeats.index,'LM']),
                                 art_vect.transform(self.feats.loc[self.metafeats.index,'Predicted'])))
        print(self.metafeats_sparse.shape)


    def get_preds(self):
        with open('../models/article_metaclassifier_xgboost.pickle','rb') as f:
            self.metaforest = pickle.load(f)
        preds = self.metaforest.predict_proba(self.metafeats_sparse)
        abs_preds = self.metaforest.predict(self.metafeats_sparse)
        final_preds = []
        #print(self.metafeats[['a','an','the','zero']])
        for l1_prob,lm_prob,meta_prob in zip(self.metafeats[['a','an','the','zero','present']].values,
                                             self.metafeats[['lm_a','lm_an','lm_the','lm_zero']].values,
                                             preds):
            #print(l1_prob,meta_prob,np.mean((l1_prob,meta_prob),axis=0))
            l1_prob[:3] *= l1_prob[-1]
            lm_prob /= sum(lm_prob)
            final_preds.append(self.options[np.argmax(np.average((l1_prob[:-1],meta_prob,lm_prob),
                                                       axis=0))])
        self.feats['Final_predicted'] = self.feats['Predicted']
        self.feats.loc[self.metafeats.index,'Final_predicted'] = final_preds
        self.feats.to_csv('test_feats.csv',sep=';',encoding='utf-8-sig')
        
        return preds
        
    def form_one_correction(self,initial,predicted,idx):
        repl = predicted+' ' if predicted != 'zero' else ''
        if any([initial.lower().startswith(x) for x in ['a ','an ','the ']]):
            predicted = self.subst.sub(repl,initial)
        else:
            if repl and initial[0] == initial[0].upper() and idx == 0:
                predicted = repl + initial[0].lower() + initial[1:]
            else:
                predicted = repl + initial
        if initial[0] == initial[0].upper() and idx == 0:
            predicted = predicted[0].upper() + predicted[1:]
        return predicted

    def form_corrections(self,preds):
        to_correct = self.feats.loc[self.feats['Final_predicted'] != self.feats['Target'],:]
        to_correct = to_correct[['Start_idx','Sent_start_idx','raw_NP','Final_predicted','Target']].values.tolist()
        idxs = []
        #print(to_correct)
        for i in range(len(to_correct)):
            initial = to_correct[i][4] if to_correct[i][4] != 'zero' else ''
            corrected = to_correct[i][3] if to_correct[i][3] != 'zero' else ''
            offset = len(initial) - len(corrected)
            new_idx = to_correct[i][0] + to_correct[i][1] + len(corrected)
            to_correct[i][3] = self.form_one_correction(to_correct[i][2],to_correct[i][3],to_correct[i][0])
            to_correct[i].pop()
            idxs.append((new_idx, offset))
            print(to_correct[i],idxs[-1])
        return to_correct,idxs

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
        corrs,idxs = self.form_corrections(preds)
        return corrs,idxs
        



# сгенерить табличку на основе текста (включая tag_sents и чанкинг)
# предсказать по табличке
# выдача - стартовый индекс в старом тексте, старая NP, новая NP
