from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk import StanfordPOSTagger
from nltk.tag import pos_tag
from nltk import RegexpParser
# from code_classifier_chunker import ConsecutiveNPChunker
import pickle
import csv
import re
from time import time
from collections import defaultdict
from conllu import parse_tree

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from feature_extraction.article_extraction import create_article_rows
from feature_extraction.preposition_extraction import create_preposition_rows

errors = {
    'Articles':
       {'csv':'articles.csv',
        'cols':['Sentence','raw_NP','NP','Start_idx','Sent_start_idx','POS_tags','Head',
             'Head_countability','NP_first_letter',
             'Head_POS','hypernyms','higher_hypernyms',
             'prevprev','prev','post','postpost','prevprev_POS','prev_POS',
             'post_POS','postpost_POS','Article'],
       'extractor':create_article_rows,
       'chunker':RegexpParser(r'NP: {<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*>}')},
    'Prepositions':
        {'csv':'prepositions.csv',
         'cols':['Sentence','raw_NP','NP','Start_idx','Sent_start_idx','POS_tags','Head',
             'Head_countability','Head_POS','hypernyms','higher_hypernyms',
             'HHead','HHead_POS','HHead_rel',
             'prev_5','prev_4','prev_3','prev_2','prev_1',
             'prev_5_POS','prev_4_POS','prev_3_POS','prev_2_POS','prev_1_POS',
             'post_1','post_2','post_3','post_4','post_5',
             'post_1_POS','post_2_POS','post_3_POS','post_4_POS','post_5_POS',
             'Preposition'],
         'extractor':create_preposition_rows,
         'chunker':RegexpParser(r'NP: {<IN>?<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*|PRP>}')}
}

error_name = 'Prepositions'
error = errors[error_name]


st = StanfordPOSTagger('../stanfordPOStagger/english-bidirectional-distsim.tagger',
                       '../stanfordPOStagger/stanford-postagger.jar',
                       java_options='-mx2048m')




#r = BNCCorpusReader(root='../../corpus/BNC/',fileids=r'B/\w*/\w*\.xml')
#tagged_sents = r.tagged_sents(c5=True)
#sents = r.sents()
#lsents = list(sents[:10000])

with open('../BNC to plain text/BNC_B_10000.txt','r',encoding='utf-8-sig') as f:
    sents = f.readlines()


sent_spans = []
token_spans = []
last_idx = -1
lim = 10000
step = 10000
i = 0
lsents = []
for sent in sents:
    sent_spans.append((last_idx+1,last_idx+1+len(sent.strip())))
    if last_idx == -1:
        last_idx = 0
    last_idx += len(sent)
    matches = [(m.group(0), (m.start(),m.end())) for m in re.finditer(r'\S+', sent)]
    tokens, idx = zip(*matches)
    lsents.append(tokens)
    token_spans.append(idx)
    i += 1
    if i == lim:
        break
    
sents = []
f = []

with open('../BNC to plain text/BNC_B_10000_parsed.txt','r',encoding='utf-8') as f:
    trees = parse_tree(f.read())

unique_words = set()

cuvplus = defaultdict(list)
with open('../cuvplus.txt','r',encoding='utf-8-sig') as f:
    for line in f.readlines():
        entry = line.strip().split('|')
        cuvplus[entry[0]].append(entry[1:])
    

a = open(error['csv'],'w',newline='',encoding='utf-8-sig')
aw = csv.writer(a,delimiter=';',quoting=csv.QUOTE_MINIMAL)
aw.writerow(error['cols'])
a.close()

#tsents = []
with open('tagged_B_'+str(lim)+'.pickle','rb') as f:
    tsents = pickle.load(f)

for j in range(step,lim+1,step):
    print(j)
    a = open(error['csv'],'a',newline='',encoding='utf-8-sig')
    aw = csv.writer(a,delimiter=';',quoting=csv.QUOTE_MINIMAL)
    curr_lsents = lsents[j-step:j]
    #curr_tsents = st.tag_sents(curr_lsents)
    #tsents.extend(curr_tsents)
    curr_tsents = tsents[j-step:j]
    curr_spans = sent_spans[j-step:j]
    curr_tok_spans = token_spans[j-step:j]
    curr_trees = trees[j-step:j]
    for sent,tsent,sent_span,token_spans,tree in \
        zip(curr_lsents,curr_tsents,curr_spans,curr_tok_spans,curr_trees):
        if ' '.join(sent) == ' '.join(sent).upper():
            sent = [x.lower() for x in sent if x]
        else:
            sent = [x for x in sent if x]
        
        for row in error['extractor'](sent,tsent,error['chunker'],cuvplus,
                                      tree,token_spans,sent_start=sent_span[0]):
            aw.writerow(row)

        #for w in sent:
        #    unique_words.add(w.lower())

    a.close()

#with open('unique_words.txt','w',encoding='utf-8') as f:
#    f.write('\n'.join(list(unique_words)))

    
#with open('tagged_'+str(lim)+'.pickle','wb') as f:
#   pickle.dump(tsents,f)
