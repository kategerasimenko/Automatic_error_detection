from nltk.corpus import wordnet as wn
from collections import defaultdict
from queue import Queue
from string import punctuation
#from get_corpus_to_trees import find_head
#from conllu import print_tree

import traceback

punct = set(punctuation) | {'‘','’','—',' ','\t','\n'}
punct_str = ''.join(punct)

class ArticleObject:
    def __init__(self,np,cuvplus,prevs,posts,idx=None,sep=';'):
        if np[0][0].lower() in {'a','an','the'}:
            self.target = np[0][0].lower()
            np = np[1:]
        else:
            self.target = 'zero'
        self.words = ' '.join([x[0] for x in np]).lower().strip(punct_str)
        self.tags = ' '.join([x[1] for x in np])
        self.head = np[-1][0].lower()
        self.first = self.words[0]
        self.head_pos = np[-1][1]
        self.countability = self.find_countability(cuvplus)
        try:
            hps = wn.synsets(self.head,pos='n')[0].hypernym_paths()
            self.hypernyms = ' '.join(list(set([x[-2].lemmas()[0].name() for x in hps])))
            self.higher_hypernyms = ' '.join(list(set([x[-(min(4,len(x)-1))].lemmas()[0].name()
                                              for x in hps if len(x) > 2])))
        except:
            self.hypernyms = ''
            self.higher_hypernyms = ''
        self.prevs = [x[0].lower() for x in prevs]
        self.prevs_pos = [x[1] for x in prevs]
        self.posts = [x[0].lower() for x in posts]
        self.posts_pos = [x[1] for x in posts]
        self.sep = sep
        self.start_idx = idx
        #k = (idxs[np[-1][0]].get()+1,np[-1][0])
        #if k in heads:
        #    self.hhead,self.hhead_pos,self.deprel = heads[k]
        #else:
        #    self.hhead,self.hhead_pos,self.deprel = '','',''

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
            #print(w)
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

def create_article_rows(sent,tsent,chunker,cuvplus,tree,spans=None,raw_sent=None,sent_start=None):
    if raw_sent is None:
        raw_sent = ' '.join(sent)
    #print(raw_sent[:20])
    full_sent = [('','')]*2 + tsent + [('','')]*2
    prevs = defaultdict(Queue)
    posts = defaultdict(Queue)
    start_idxs = defaultdict(Queue)
    end_idxs = defaultdict(Queue)
    if len(sent) != len(tsent):
        print(sent,tsent,sent_start)
        return None
    for i in range(len(sent)):
        if spans is not None and tsent[i][0]:
            start_idxs[tsent[i][0]].put(spans[i][0])
            end_idxs[tsent[i][0]].put(spans[i][1])
        prevs[full_sent[i+2][0]].put(full_sent[i:i+2])
        posts[full_sent[i+2][0]].put(full_sent[i+3:i+5])
    try:
        chunks = chunker.parse(tsent)
    except ValueError:
        print('sent not parsed',tsent)
        return None
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
            if subtree and subtree[-1][1].startswith('NN'):
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
                        for w in subtree[1:]:
                            start_idxs[w[0]].get()
                        for w in subtree[:-1]:
                            end_idxs[w[0]].get()
                    else:
                        start_idx,end_idx = None,None
                        init_words = ' '.join([x[0] for x in subtree])
                    obj = ArticleObject(subtree,cuvplus,curr_prevs,curr_posts,start_idx)
                    yield [raw_sent,init_words,obj.words,obj.start_idx,sent_start,
                           obj.tags,obj.head,obj.countability,obj.first,
                           obj.head_pos,obj.hypernyms,obj.higher_hypernyms] + \
                           obj.prevs + obj.prevs_pos + obj.posts + obj.posts_pos + \
                           [obj.target]
                except:
                    pass
                    #traceback.print_exc()
                    #print('smth wrong:',chunks,'\n',subtree)
            else:
                prevs,posts,start_idxs,end_idxs = maintain_queues(subtree,prevs,posts,start_idxs,end_idxs,spans)
        else:
            prevs,posts,start_idxs,end_idxs = maintain_queues(subtree,prevs,posts,start_idxs,end_idxs,spans)
