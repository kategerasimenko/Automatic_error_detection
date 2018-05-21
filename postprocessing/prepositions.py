import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix, lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from nltk.tokenize import TreebankWordTokenizer
from nltk import RegexpParser
from collections import defaultdict
import re
import json


import sys, os
sys.path.insert(0, os.path.abspath('..'))
from feature_extraction.preposition_extraction import create_preposition_rows
from feature_extraction.lm_probas import get_lm_probas
from preprocessing.conllu import parse_tree


class PrepositionCorrector:
    def __init__(self):
        grammar = r'NP: {<IN>?<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*|PRP>}'
        self.chunker = RegexpParser(grammar)
        self.tokenizer = TreebankWordTokenizer()
        with open('../prepositions.txt','r',encoding='utf-8-sig') as f:
            self.options = f.read().split('\n')
            self.options.append('zero')
            self.options_for_lookup = set(self.options[:-1])

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



    def get_table(self,sents,tsents,idxs,raw_sents,sent_spans):
        with open('init_sents_for_prepositions_test_parsed.txt','r',encoding='utf-8') as f:
            trees = parse_tree(f.read())
        for sent,tsent,spans,raw_sent,sent_span,tree in zip(sents,tsents,idxs,raw_sents,sent_spans,trees):
            if ' '.join(sent) == ' '.join(sent).upper():
                sent = [x.lower() for x in sent if x]
            else:
                sent = [x for x in sent if x]
            self.feats.extend(create_preposition_rows(sent,tsent,self.chunker,self.cuvplus,
                                                      tree,spans,raw_sent,sent_span[0]))
        self.feats = pd.DataFrame(self.feats,columns=self.cols)
        

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
            

        all_vectors = lil_matrix((self.feats.shape[0],300))
        for i,word in enumerate(self.feats['Head']):
            if word in w2v_model:
               all_vectors[i,:] = w2v_model[word]
 
        all_vectors_hhead = lil_matrix((self.feats.shape[0],300))
        for i,word in enumerate(self.feats['Head']):
            if word in w2v_model:
               all_vectors_hhead[i,:] = w2v_model[word]

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

        self.data_sparse = hstack((npm,pos,head,countability,head_pos,hyp,hhyp,all_vectors,
                                   hhead,hhead_pos,deprel,all_vectors_hhead,
                                   prevs,prevs_pos,posts,posts_pos)).tocsr()


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


    def normalize_preposition_corr(self,corr,position):
        # position - before, NP, over
        #print(corr,position)
        if corr == 'DELETE':
            return 'zero'
        toks = corr.split()
        if position == 'NP':
            if toks[0].lower() in self.options_for_lookup:
                return toks[0].lower()
            else:
                return 'zero'
        elif position == 'before':
            if toks[-1].lower() in self.options_for_lookup:
                return toks[-1].lower()
            else:
                return 'zero'
        else:
            toks_in_preps = [i for i,x in enumerate(toks) 
                             if x.lower() in self.options_for_lookup]
            if len(toks_in_preps) > 1:
                return None
            elif not toks_in_preps:
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
                             self.normalize_preposition_corr(error_spans[i][2],'np')]
            else:
                corr = self.normalize_preposition_corr(error_spans[i][2],'over')
                if corr is None:
                    corr = np.Target
                return i+1, [np.raw_NP,np.Start_idx,np.Sent_start_idx,np.Target,np.Predicted,corr]
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
        # probs = get_lm_probas(str_sents)
        # temp for local windows - file is processed by lm on server separately
        #self.write_sentences_for_lm(str_sents)
        with open('lm_preds_test_prepositions.json','r',encoding='utf-8') as f:
            probs = json.loads(f.read())
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
        self.metafeats_sparse = hstack((self.metafeats.drop(['lm_'+x for x in self.options],axis=1).to_sparse(),
                                 art_vect.transform(self.feats.loc[self.metafeats.index,'Target']),
                                 art_vect.transform(self.feats.loc[self.metafeats.index,'LM']),
                                 art_vect.transform(self.feats.loc[self.metafeats.index,'Predicted'])))
        print(self.metafeats_sparse.shape)


    def get_preds(self):
        with open('../models/preposition_metaclassifier_xgboost.pickle','rb') as f:
            self.metaforest = pickle.load(f)
        preds = self.metaforest.predict_proba(self.metafeats_sparse)
        abs_preds = self.metaforest.predict(self.metafeats_sparse)
        final_preds = []
        preds = pd.DataFrame(preds,columns=self.metaforest.classes_)
        option_interj = sorted(list(set(self.logit_type.classes_) & set(self.metaforest.classes_)))
        for l1_prob,lm_prob,meta_prob in zip(self.metafeats[option_interj+['present']].values,
                                             self.metafeats[['lm_'+x for x in option_interj]].values,
                                             preds[option_interj].values):            
            #print(l1_prob,meta_prob,np.mean((l1_prob,meta_prob),axis=0))
            l1_prob[:-2] *= l1_prob[-1]
            final_preds.append(option_interj[np.argmax(np.average((l1_prob[:-1],meta_prob,lm_prob),
                                                       weights=(0.25,0.5,0.25),axis=0))])            
        self.feats['Final_predicted'] = self.feats['Predicted']
        self.feats.loc[self.metafeats.index,'Final_predicted'] = final_preds
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
        to_correct = to_correct[['Start_idx','Sent_start_idx','raw_NP','Final_predicted']].values.tolist()
        #print(to_correct)
        for i in range(len(to_correct)):
            to_correct[i][3] = self.form_one_correction(to_correct[i][2],to_correct[i][3],to_correct[i][0])
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
# выдача - стартовый индекс в старом тексте, старая NP, новая NP
