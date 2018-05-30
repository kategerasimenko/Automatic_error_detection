from nltk import RegexpParser
import pickle
import csv
import re
from time import time
from collections import defaultdict
from article_extraction import create_article_rows

errors = {
    'Articles':
       {'csv':'articles.csv',
        'cols':['Sentence','raw_NP','NP','Start_idx','Sent_start_idx','POS_tags','Head',
             'Head_countability','NP_first_letter',
             'Head_POS','hypernyms','higher_hypernyms',
             'prev_2','prev_1','prev_2_POS','prev_1_POS',
             'post_1','post_2','post_1_POS','post_2_POS',
             'Article'],
       'extractor':create_article_rows,
       'chunker':RegexpParser(r'NP: {<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*>}')}
}

error_name = 'Articles'
error = errors[error_name]


#with open('../../kenlm/BNC_plain.txt','r',encoding='utf-8-sig') as f:
#    sents = f.read().split('\n')
#    sents = [x for x in sents if x.strip()]

#with open('../BNC to plain text/BNC_B_10000_parsed.txt','r',encoding='utf-8') as f:
#    trees = parse_tree(f.read())

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
with open('../parse_BNC/BNC_tagged.txt','r') as f:
    tsents = f.read().split('\n')
    
lim = len(tsents)
print(lim)
step = 100000
last_idx = -1
border_sent = False

for j in range(2400000,lim+1,step):
    print(j)
    a = open(error['csv'],'a',newline='',encoding='utf-8-sig')
    aw = csv.writer(a,delimiter=';',quoting=csv.QUOTE_MINIMAL)
    curr_tsents = [[tuple(x.rsplit('_',1)) for x in sent.split(' ')] for sent in tsents[j-step:j]]
    curr_lsents = []
    curr_spans = []
    curr_tok_spans = []
    i = -1
    final_idx = 0
    for tsent in curr_tsents:
        i += 1
        sent = ' '.join([x[0] for x in tsent])
        if border_sent:
            curr_spans.append((last_idx+1,last_idx+1+len(sent.strip())))
            if last_idx == -1:
                last_idx = 0
            last_idx += len(sent)
            matches = [(m.group(0), (m.start(),m.end())) for m in re.finditer(r'\S+', sent)]
            if matches:
              tokens, idx = zip(*matches)
            #else:
            #  print(last_idx,sent)
            #  tokens,idx = [],(0,0)
              curr_lsents.append(tokens)
              curr_tok_spans.append(idx)
        elif sent == 'We have seen pairs of words ( a , b ) interpreted as double-precision arithmetic formats , with corresponding operations .':
            border_sent = True
            final_idx = i+1
            print('border sent found')  
    #curr_tsents = st.tag_sents(curr_lsents)
    #tsents.extend(curr_tsents)
    #curr_trees = trees[j-step:j]
    curr_trees = [None]*len(curr_lsents)
    if final_idx:
        curr_tsents = curr_tsents[final_idx:]
    for sent,tsent,sent_span,token_spans,tree in \
        zip(curr_lsents,curr_tsents,curr_spans,curr_tok_spans,curr_trees):
        if ' '.join(sent) == ' '.join(sent).upper():
            sent = [x.lower() for x in sent if x]
        else:
            sent = [x for x in sent if x]
        if sent:
          row_gen = error['extractor'](sent,tsent,error['chunker'],cuvplus,
                                        tree,token_spans,sent_start=sent_span[0])
          if row_gen is not None:
              for row in row_gen:
                  aw.writerow(row)

          for w in sent:
              unique_words.add(w.lower())

    a.close()

with open('unique_words.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(list(unique_words)))

    
#with open('tagged_'+str(lim)+'.pickle','wb') as f:
#   pickle.dump(tsents,f)
