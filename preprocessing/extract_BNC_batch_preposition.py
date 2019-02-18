from nltk import RegexpParser
import pickle
import csv
import re
from time import time
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from feature_extraction.preposition_extraction import create_preposition_rows
from conllu import parse_tree


def parsed_sent_generator(path):
    sent = ''
    tokens = []
    tagged_tokens = []
    token_spans = []
    sent_start = 0
    last_idx = 0
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                yield sent,tokens,tagged_tokens,token_spans,\
                      (sent_start,sent_start+last_idx)
                sent = ''
                tokens = []
                tagged_tokens = []
                token_spans = []
                sent_start = 0
                last_idx = 0
            elif '\t' in line:
                sent += line
                cells = line.split('\t')
                tokens.append(cells[1])
                raw_span = cells[-1].split('|')[-1].split('=')[1].split(':')
                if not sent_start:
                    sent_start = int(raw_span[0])
                span = [int(x)-sent_start for x in raw_span]
                token_spans.append(tuple(span))
                tagged_tokens.append((cells[1],cells[4]))
                last_idx += span[1]
        yield sent,tokens,tagged_tokens,token_spans,\
              (sent_start,sent_start+last_idx)


errors = {
    'Prepositions':
        {'csv':'prepositions.csv',
         'cols':['NP','POS_tags','Head',
             'Head_countability','Head_POS','hypernyms','higher_hypernyms',
             'HHead','HHead_POS','HHead_rel',
             'prev_5','prev_4','prev_3','prev_2','prev_1',
             'prev_5_POS','prev_4_POS','prev_3_POS','prev_2_POS','prev_1_POS',
             'post_1','post_2','post_3','post_4','post_5',
             'post_1_POS','post_2_POS','post_3_POS','post_4_POS','post_5_POS',
             'Preposition'],
         'extractor':create_preposition_rows,
         'chunker':RegexpParser(r'NP: {<IN|TO>?<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*|PRP>}')}
}

error_name = 'Prepositions'
error = errors[error_name]


#with open('../../kenlm/BNC_plain.txt','r',encoding='utf-8-sig') as f:
#    sents = f.read().split('\n')
#    sents = [x for x in sents if x.strip()]

#with open('../BNC to plain text/BNC_B_10000_parsed.txt','r',encoding='utf-8') as f:
#    trees = parse_tree(f.read())


cuvplus = defaultdict(list)
with open('cuvplus.txt','r',encoding='utf-8-sig') as f:
    for line in f.readlines():
        entry = line.strip().split('|')
        cuvplus[entry[0]].append(entry[1:])
    

#a = open(error['csv'],'w',newline='',encoding='utf-8-sig')
#aw = csv.writer(a,delimiter=';',quoting=csv.QUOTE_MINIMAL)
#aw.writerow(error['cols'])
#a.close()

    
lim = 100
print(lim)
start = 0
step = 100000
last_idx = -1
border_sent = True
curr_trees = []
rows = []
open(error['csv'],'w',newline='',encoding='utf-8-sig').close()

for j,conllu_info in enumerate(parsed_sent_generator('../feature_extraction/test_parsed.txt')):
    try:
        if j > lim:
            break
        if j < start:
            continue
        if not j % step and j != start:
            print(j)
            a = open(error['csv'],'a',newline='',encoding='utf-8-sig')
            aw = csv.writer(a,delimiter=';',quoting=csv.QUOTE_MINIMAL)
            aw.writerows(rows)
            rows = []
            a.close()
        tree,sent,tsent,token_spans,sent_span = conllu_info
        if tree.strip():
            tree = parse_tree(tree)[0]
            if ' '.join(sent) == ' '.join(sent).upper():
                sent = [x.lower() for x in sent if x]
            else:
                sent = [x for x in sent if x]
            if sent:
                row_gen = error['extractor'](sent,tsent,error['chunker'],cuvplus,
                                             tree,token_spans,sent_start=sent_span[0])
                if row_gen is not None:
                    for row in row_gen:
                        rows.append(row)
    except:
        print('Error has occured at line', j)
        traceback.print_exc()
        continue

a = open(error['csv'],'a',newline='',encoding='utf-8-sig')
aw = csv.writer(a,delimiter=';',quoting=csv.QUOTE_MINIMAL)
aw.writerows(rows)
a.close()

    
#with open('tagged_'+str(lim)+'.pickle','wb') as f:
#   pickle.dump(tsents,f)
