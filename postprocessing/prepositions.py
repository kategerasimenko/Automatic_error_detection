import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix, lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import minmax_scale
from xgboost import XGBClassifier
from nltk.tokenize import TreebankWordTokenizer
from nltk import RegexpParser
from collections import defaultdict
import re
import json


import sys, os
sys.path.insert(0, os.path.abspath('..'))
from preposition_table.preposition_extraction import create_preposition_rows
from lm_probas import get_lm_probas
from preposition_table.conllu import parse_tree


class PrepositionCorrector:
    def __init__(self):
        grammar = r'NP: {<IN|TO>?<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*|PRP>}'
        self.chunker = RegexpParser(grammar)
        self.tokenizer = TreebankWordTokenizer()
        with open('../prepositions.txt','r',encoding='utf-8-sig') as f:
            self.options = f.read().split('\n')
            self.options.append('zero')
            self.options_for_lookup = set(self.options[:-1])

        with open('../all_prepositions.txt','r',encoding='utf-8') as f:
            self.all_preps = set(f.read().split('\n'))

        self.cuvplus = defaultdict(list)
        with open('../cuvplus.txt','r',encoding='utf-8-sig') as f:
            for line in f.readlines():
                entry = line.strip().split('|')
                self.cuvplus[entry[0]].append(entry[1:])

        self.feats = []
        
        self.cols = ['Sentence','raw_NP','NP','Start_idx','Sent_start_idx','POS_tags','Head',
             'Head_countability','Head_POS','hypernyms','higher_hypernyms',
             'HHead','HHead_POS','HHead_rel',
             'prev_5','prev_4','prev_3','prev_2','prev_1',
             'prev_5_POS','prev_4_POS','prev_3_POS','prev_2_POS','prev_1_POS',
             'post_1','post_2','post_3','post_4','post_5',
             'post_1_POS','post_2_POS','post_3_POS','post_4_POS','post_5_POS',
             'Target']

        with open('../models/preposition_logit_binary.pickle','rb') as f:
           self.logit_bin = pickle.load(f)
            
        with open('../models/preposition_logit_type.pickle','rb') as f:
           self.logit_type = pickle.load(f)



    def get_table(self,sents,tsents,idxs,raw_sents,sent_spans,trees):
        for sent,tsent,spans,raw_sent,sent_span,tree in zip(sents,tsents,idxs,raw_sents,sent_spans,trees):
            if ' '.join(sent) == ' '.join(sent).upper():
                sent = [x.lower() for x in sent if x]
            else:
                sent = [x for x in sent if x]
            self.feats.extend(create_preposition_rows(sent,tsent,self.chunker,self.cuvplus,
                                                      tree,spans,raw_sent,sent_span[0]))
        self.feats = pd.DataFrame(self.feats,columns=self.cols)
        self.feats['HHead'].replace('', np.nan, inplace=True)
        self.feats.dropna(subset=['HHead'],inplace=True)
        

    def get_feature_matrix(self,w2v_model):
        with open('../models/one_word_vectorizer.pickle','rb') as f:
           onewordvect = pickle.load(f)

        with open('../models/pos_vectorizer.pickle','rb') as f:
            pos_vect = pickle.load(f)

        with open('../models/extended_np_vectorizer.pickle','rb') as f:
           np_vect = pickle.load(f)
            
        with open('../models/noun_hypernym_vectorizer.pickle','rb') as f:
           hyp_vect = pickle.load(f)
            
        with open('../models/noun_higher_hypernym_vectorizer.pickle','rb') as f:
           hhyp_vect = pickle.load(f)

        with open('../models/countability_vectorizer.pickle','rb') as f:
           count_vect = pickle.load(f)
 
        with open('../models/hhead_vectorizer.pickle','rb') as f:
            hhead_vect = pickle.load(f)
            
        with open('../models/deprel_vectorizer.pickle','rb') as f:
            deprel_vect = pickle.load(f)
        

        npm = np_vect.transform(self.feats['NP'])

        pos = pos_vect.transform(self.feats['POS_tags'])
        head_pos = pos_vect.transform(self.feats['Head_POS'])
        hhead_pos = pos_vect.transform(self.feats['HHead_POS'])
        prevs_pos = hstack([pos_vect.transform(self.feats['prev_'+str(i)+'_POS']) for i in range(1,6)])
        posts_pos = hstack([pos_vect.transform(self.feats['post_'+str(i)+'_POS']) for i in range(1,6)])

        countability = count_vect.transform(self.feats['Head_countability'])

        hyp = hyp_vect.transform(self.feats['hypernyms'])
        hhyp = hhyp_vect.transform(self.feats['higher_hypernyms'])

        deprel = deprel_vect.transform(self.feats['HHead_rel'])
        hhead = hhead_vect.transform(self.feats['HHead'])

        head = onewordvect.transform(self.feats['Head'])
        prevs = hstack([onewordvect.transform(self.feats['prev_'+str(i)]) for i in range(1,6)])
        posts = hstack([onewordvect.transform(self.feats['post_'+str(i)]) for i in range(1,6)])

        self.data_sparse = hstack((npm,pos,head,countability,head_pos,hyp,hhyp,
                                   hhead,hhead_pos,deprel,
                                   prevs,prevs_pos,posts,posts_pos)).tocsr()
        with open('../models/preposition_nonzero_columns.pickle','rb') as f:
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


    def normalize_preposition_corr(self,corr,position):
        # position - before, NP, over
        if corr == 'DELETE':
            return 'zero'
        toks = corr.split()
        if position == 'NP':
            if toks[0].lower() in self.options_for_lookup:
                return toks[0].lower()
            elif toks[0].lower() in self.all_preps:
                return None
            else:
                return 'zero'
        elif position == 'before':
            if toks[-1].lower() in self.options_for_lookup:
                return toks[-1].lower()
            elif toks[-1].lower() in self.all_preps:
                return None
            else:
                return 'zero'
        else:
            toks_in_preps = [i for i,x in enumerate(toks) 
                             if x.lower() in self.options_for_lookup]
            toks_in_all_preps = [i for i,x in enumerate(toks) 
                             if x.lower() in self.all_preps]
            if len(toks_in_preps) > 1:
                return None
            elif not toks_in_preps:
                if toks_in_all_preps:
                    return None
                else:
                    return 'zero'
            else:
                return toks[toks_in_preps[0]].lower()

    def get_error_span(self,np,i,error_spans):
        '''
        three situations:
            preposition marked before np: [depend of] age
            preposition marked as np or at the beginning: depend [of age] or depend [of] age
            preposition marked over everything: [depend of age]
        '''
        while i < len(error_spans) and \
              error_spans[i][1] < np.Sent_start_idx + np.Start_idx - 1:
            i += 1
        if i < len(error_spans) and error_spans[i][1] == np.Sent_start_idx + np.Start_idx - 1:
            corr = self.normalize_preposition_corr(error_spans[i][2],'before')
            if corr == 'zero':
                return i, [np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Target,np.Predicted,np.Target]
            else:
                return i+1, [np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Target,np.Predicted,corr]
        elif i < len(error_spans) and (error_spans[i][1] == np.Sent_start_idx + len(np.raw_NP) + np.Start_idx \
                                       or error_spans[i][0] == np.Sent_start_idx + np.Start_idx):
            if error_spans[i][0] == np.Sent_start_idx + np.Start_idx:
                return i+1, [np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Target,np.Predicted,
                             self.normalize_preposition_corr(error_spans[i][2],'NP')]
            elif error_spans[i][0] < np.Sent_start_idx + np.Start_idx:
                corr = self.normalize_preposition_corr(error_spans[i][2],'over')
                return i+1, [np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Target,np.Predicted,corr]
            else:
                return i+1,[np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Target,np.Predicted,np.Target]
                
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
        with open('test_sents_prepositions.txt','w',encoding='utf-8') as f:
            f.write('\n\n'.join(['\n'.join(x) for x in sents]))
            f.write('\n')


    def get_metamatrix(self):
        preds, preds_type = self.get_probas()
        str_sents = self.get_sentences_for_lm()
        probs = get_lm_probas(str_sents)
        # temp for local windows - file is processed by lm on server separately
        #self.write_sentences_for_lm(str_sents)
        #with open('lm_preds_test_prepositions.json','r',encoding='utf-8') as f:
        #    probs = json.loads(f.read())
        # end temp
        self.metafeats = pd.concat([pd.DataFrame(preds,columns=['present','zero']),
                                    pd.DataFrame(preds_type,columns=self.logit_type.classes_),
                                    pd.DataFrame(probs,columns=['lm_'+x for x in self.options])],
                                   axis=1)
        probs_ratio = []
        probs_delta = []
        init_probs = []
        corr_probs = []
        lm_choice = []
        print(self.metafeats.shape[0],self.feats.shape[0])
        print(pd.DataFrame(preds,columns=['present','zero']).shape,
                                    pd.DataFrame(preds_type,columns=self.logit_type.classes_).shape,
                                    pd.DataFrame(probs,columns=['lm_'+x for x in self.options]).shape)
        for i in range(self.metafeats.shape[0]):
            row = self.metafeats.iloc[i]
            feat_row = self.feats.iloc[i]
            init_prob = row['lm_'+feat_row['Target']]
            corr_prob = row['lm_'+feat_row['Predicted']]
            init_probs.append(init_prob)
            corr_probs.append(corr_prob)
            probs_ratio.append(init_prob / corr_prob)
            probs_delta.append(init_prob - corr_prob)
            lm_choice.append(np.argmax(row[['lm_'+x for x in self.options]]).split('_')[1])
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

        with open('../models/preposition_choice_vectorizer.pickle','rb') as f:
            art_vect = pickle.load(f)
        self.metafeats_sparse = hstack((self.metafeats.to_sparse(),
                                 art_vect.transform(self.feats.loc[self.metafeats.index,'Target']),
                                 art_vect.transform(self.feats.loc[self.metafeats.index,'LM']),
                                 art_vect.transform(self.feats.loc[self.metafeats.index,'Predicted'])))
        print(self.metafeats_sparse.shape)


    def get_preds(self):
        with open('../models/preposition_metaclassifier_logit.pickle','rb') as f:
            self.metaforest = pickle.load(f)
        preds = self.metaforest.predict_proba(self.metafeats_sparse)
        abs_preds = self.metaforest.predict(self.metafeats_sparse)
        final_preds = []
        probs = []
        for l1_prob,lm_prob,meta_prob in zip(self.metafeats[self.options+['present']].values,
                                             self.metafeats[['lm_'+x for x in self.options]].values,
                                             preds):            
            #print(l1_prob,meta_prob,np.mean((l1_prob,meta_prob),axis=0))
            l1_prob[:-2] *= l1_prob[-1]
            lm_prob /= sum(lm_prob)
            final_preds.append(self.options[np.argmax(np.average((l1_prob[:-1],meta_prob,lm_prob),
                                                       weights=[0.25,0.5,0.25],axis=0))])
            probs.append(np.max(np.average((l1_prob[:-1],meta_prob,lm_prob),
                                                       weights=[0.25,0.5,0.25],axis=0)))
        self.feats['Final_predicted'] = self.feats['Predicted']
        self.feats['probs'] = 0
        self.feats.loc[self.metafeats.index,'Final_predicted'] = abs_preds
        self.feats.loc[self.metafeats.index,'probs'] = probs
        self.feats.to_csv('test_feats_preps.csv',sep=';',encoding='utf-8-sig')
        
        return preds
        
    def form_one_correction(self,initial,predicted,idx):
        corrected = False
        repl = predicted+' ' if predicted != 'zero' else ''
        for x in self.options[:-1]:
            if initial.lower().startswith(x+' '):
                predicted = repl + initial[len(x)+1:]
                corrected = True
                break
        if not corrected:
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
        return to_correct,idxs

    def detect_errors(self,w2v_model,sents,tsents,idxs,raw_sents,sent_spans):
        self.sents = sents
        self.tsents = tsents
        self.idx_to_sent = {idx[0]:sent for idx,sent in zip(sent_spans,sents)}
        print('\tGetting table')
        with open('init_sents_for_prepositions_test_parsed.txt','r',encoding='utf-8') as f:
            trees = parse_tree(f.read())
        self.get_table(sents,tsents,idxs,raw_sents,sent_spans,trees)
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
