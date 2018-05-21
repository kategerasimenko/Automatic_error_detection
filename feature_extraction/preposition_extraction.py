from nltk.corpus import wordnet as wn
from collections import defaultdict
from queue import Queue
from string import punctuation

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from preprocessing.get_corpus_to_trees import find_head

import traceback

punct = set(punctuation) | {'‘','’','—',' ','\t','\n'}
punct_str = ''.join(punct)

#preps from Dahlmeier & Ng 2011 ASO
with open('../prepositions.txt','r',encoding='utf-8-sig') as f:
    preposition_set = {x.strip() for x in f.readlines()}

class PrepositionObject:
    def __init__(self,np,cuvplus,prevs,posts,heads,idx=None,
                 head_start_idx=None,sep=';',sent_start=None):
        if np[0][1] == 'IN':
            self.target = np[0][0].lower()
            np = np[1:]
        else:
            self.target = 'zero'
        self.words = ' '.join([x[0] for x in np]).lower().strip(punct_str)
        self.tags = ' '.join([x[1] for x in np])
        self.head = np[-1][0].lower()
        self.head_pos = np[-1][1]
        self.countability = self.find_countability(cuvplus)
        if self.head_pos != 'PRP':
            try:
                hps = wn.synsets(self.head,pos='n')[0].hypernym_paths()
                self.hypernyms = ' '.join(list(set([x[-2].lemmas()[0].name() for x in hps])))
                self.higher_hypernyms = ' '.join(list(set([x[-(min(4,len(x)-1))].lemmas()[0].name()
                                                  for x in hps if len(x) > 2])))
            except:
                self.hypernyms = ''
                self.higher_hypernyms = ''
        else:
            self.hypernyms = ''
            self.higher_hypernyms = ''        
        self.prevs = [x[0].lower() for x in prevs]
        self.prevs_pos = [x[1] for x in prevs]
        self.posts = [x[0].lower() for x in posts]
        self.posts_pos = [x[1] for x in posts]
        self.sep = sep
        self.start_idx = idx
        if head_start_idx is not None and sent_start is not None:
            full_idx = head_start_idx + sent_start
        else:
            full_idx = None
        k = (full_idx,np[-1][0])
        #print(heads,k,sent_start)
        if k in heads:
            self.hhead,self.hhead_pos,self.deprel = heads[k]
        else:
            self.hhead,self.hhead_pos,self.deprel = '','',''

    def find_countability(self,cuvplus):
        if self.head in cuvplus:
            occurrences = cuvplus[self.head]
        else:
            return ''
        mapping = {'K':'C','L':'U','M':'both','N':'proper'}
        pos = [mapping[y[0]] for x in occurrences for y in x[2].split(',')
               if y[0] in mapping]
        if pos:
            return pos[0]
        else:
            return ''
        
def maintain_queues(subtree,prevs,posts,start_idxs,end_idxs,spans):
    for w in subtree:
        if isinstance(w,tuple):
            prevs[w[0]].get()
            posts[w[0]].get()
            if spans is not None:
                start_idxs[w[0]].get()
                end_idxs[w[0]].get()
        else: # if there are trees put their words to the end of the queue
              # as they will be processed later
            for subw in w:
                prevs[subw[0]].put(prevs[subw[0]].get())
                posts[subw[0]].put(posts[subw[0]].get())
                if spans is not None:
                    start_idxs[subw[0]].put(start_idxs[subw[0]].get())
                    end_idxs[subw[0]].put(end_idxs[subw[0]].get())
    return prevs,posts,start_idxs,end_idxs

def create_preposition_rows(sent,tsent,chunker,cuvplus,tree,spans=None,raw_sent=None,sent_start=None):
    if raw_sent is None:
        raw_sent = ' '.join(sent)
    #print(sent,spans,sent_start)
    full_sent = [('','')]*5 + tsent + [('','')]*5
    prevs = defaultdict(Queue)
    posts = defaultdict(Queue)
    start_idxs = defaultdict(Queue)
    end_idxs = defaultdict(Queue)
    heads = find_head(tree,lambda x: x.startswith('N') or x == 'PRP')
    #print(heads)
    for i in range(len(sent)):
        if spans is not None and tsent[i][0]:
            start_idxs[tsent[i][0]].put(spans[i][0])
            end_idxs[tsent[i][0]].put(spans[i][1])
        prevs[full_sent[i+5][0]].put(full_sent[i:i+5])
        posts[full_sent[i+5][0]].put(full_sent[i+6:i+11])
    chunks = chunker.parse(tsent)
    for subtree in chunks.subtrees():
        if subtree.label() == 'NP':
            subtree = list(subtree)
            while subtree and subtree[-1][0] in punct:
                prevs[subtree[-1][0]].get()
                posts[subtree[-1][0]].get()
                if spans is not None:
                    start_idxs[subtree[-1][0]].get()
                    end_idxs[subtree[-1][0]].get()
                subtree.pop()
            while subtree and subtree[0][0] in punct:
                prevs[subtree[0][0]].get()
                posts[subtree[0][0]].get()
                if spans is not None:
                    start_idxs[subtree[0][0]].get()
                    end_idxs[subtree[0][0]].get()
                subtree.pop(0)
            if subtree and (subtree[-1][1].startswith('NN') or subtree[-1][1] == 'PRP') and \
               (subtree[0][0].lower() in preposition_set or subtree[0][1] != 'IN'):
                try:
                    curr_prevs = prevs[subtree[0][0]].get()
                    curr_posts = posts[subtree[-1][0]].get()
                    for w in subtree[1:]:
                        prevs[w[0]].get()
                    for w in subtree[:-1]:
                        posts[w[0]].get()                    
                    if spans is not None:
                        start_idx = start_idxs[subtree[0][0]].get()
                        end_idx = end_idxs[subtree[-1][0]].get()
                        init_words = raw_sent[start_idx:end_idx]
                        for w in subtree[1:-1]:
                            start_idxs[w[0]].get()
                        for w in subtree[:-1]:
                            end_idxs[w[0]].get()
                        if len(subtree) > 1:
                            head_start_idx = start_idxs[subtree[-1][0]].get()
                        else:
                            head_start_idx = start_idx
                    else:
                        start_idx,end_idx = None,None
                        init_words = ' '.join([x[0] for x in subtree])
                    obj = PrepositionObject(subtree,cuvplus,curr_prevs,curr_posts,heads,start_idx,
                                            head_start_idx,sent_start=sent_start)
                    yield [raw_sent,init_words,obj.words,obj.start_idx,sent_start,
                           obj.tags,obj.head,obj.countability,
                           obj.head_pos,obj.hypernyms,obj.higher_hypernyms,
                           obj.hhead,obj.hhead_pos,obj.deprel] + \
                           obj.prevs + obj.prevs_pos + obj.posts + obj.posts_pos + [obj.target]
                except:
                    traceback.print_exc()
                    print('smth wrong:',chunks,'\n',subtree)
            else:
                prevs,posts,start_idxs,end_idxs = maintain_queues(subtree,prevs,posts,start_idxs,end_idxs,spans)
        else:
            prevs,posts,start_idxs,end_idxs = maintain_queues(subtree,prevs,posts,start_idxs,end_idxs,spans)
