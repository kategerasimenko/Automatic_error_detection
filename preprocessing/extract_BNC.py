from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk import StanfordPOSTagger
from nltk.tag import pos_tag
from nltk import RegexpParser
# from code_classifier_chunker import ConsecutiveNPChunker
from article_extraction import create_article_rows
# import pickle
import csv
from time import time
from collections import defaultdict
from conllu import parse_tree


st = StanfordPOSTagger('../stanfordPOStagger/english-bidirectional-distsim.tagger',
                       '../stanfordPOStagger/stanford-postagger.jar')



# classifier-based chunker - not really cool
# with open('chunker.pickle', 'rb') as f:
#    chunker = pickle.load(f)

#regexp-based chunker
grammar = r'NP: {<DT|JJ.?|PRP\$|POS|RB.|CD|NN.*>*<NN.*>}'
chunker = RegexpParser(grammar)

r = BNCCorpusReader(root='../../corpus/BNC/',fileids=r'B/\w*/\w*\.xml')
#tagged_sents = r.tagged_sents(c5=True)
sents = r.sents()

with open('../BNC to plain text/BNC_B_10000_parsed.txt','r',encoding='utf-8') as f:
    trees = parse_tree(f.read())

unique_words = set()

cuvplus = defaultdict(list)
with open('cuvplus.txt','r',encoding='utf-8-sig') as f:
    for line in f.readlines():
        entry = line.strip().split('|')
        cuvplus[entry[0]].append(entry[1:])
    

a = open('articles.csv','w',newline='',encoding='utf-8-sig')
aw = csv.writer(a,delimiter=';',quoting=csv.QUOTE_MINIMAL)
aw.writerow(['NP','POS_tags','Head','Head_countability','NP_first_letter',
             'Head_POS','hypernyms','higher_hypernyms','hhead','hhead_POS','deprel',
             'prevprev','prev','post','postpost','prevprev_POS','prev_POS',
             'post_POS','postpost_POS','Article'])

lsents = list(sents[:10000])
tsents = st.tag_sents(lsents)
for i,sent in enumerate(lsents):
    if not i % 1000:
        print (i)

    if ' '.join(sent) == ' '.join(sent).upper():
        sent = [x.lower() for x in sent if x]
    else:
        sent = [x for x in sent if x]

    tsent = tsents[i]
    tree = trees[i]
    
    for row in create_article_rows(sent,tsent,tree,chunker,cuvplus):
        aw.writerow(row)

    for w in sent:
        unique_words.add(w.lower())

a.close()

with open('unique_words.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(list(unique_words)))
