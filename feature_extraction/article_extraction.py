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
    def __init__(self,np,cuvplus,prevprev,prev,post,postpost,idx=None,sep=';'):
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
        self.prevprev = prevprev[0].lower()
        self.prev = prev[0].lower()
        self.post = post[0].lower()
        self.postpost = postpost[0].lower()
        self.prevprev_pos = prevprev[1]
        self.prev_pos = prev[1]
        self.post_pos = post[1]
        self.postpost_pos = postpost[1]
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

def create_article_rows(sent,tsent,chunker,cuvplus,spans=None,raw_sent=None,sent_start=None):
    if raw_sent is None:
        raw_sent = ' '.join(sent)
    #print(raw_sent[:20])
    full_sent = [('',''),('','')] + tsent + [('',''),('','')]
    prevs = defaultdict(Queue)
    posts = defaultdict(Queue)
    start_idxs = defaultdict(Queue)
    end_idxs = defaultdict(Queue)
    for i in range(len(sent)):
        if spans is not None and tsent[i][0]:
            start_idxs[tsent[i][0]].put(spans[i][0])
            end_idxs[tsent[i][0]].put(spans[i][1])
        prevs[full_sent[i+2][0]].put((full_sent[i],full_sent[i+1]))
        posts[full_sent[i+2][0]].put((full_sent[i+3],full_sent[i+4]))
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
            if subtree and subtree[-1][1].startswith('NN'):
                try:
                    prevprev,prev = prevs[subtree[0][0]].get()
                    post,postpost = posts[subtree[-1][0]].get()
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
                    obj = ArticleObject(subtree,cuvplus,prevprev,prev,post,postpost,start_idx)
                    yield [raw_sent,init_words,obj.words,obj.start_idx,sent_start,
                           obj.tags,obj.head,obj.countability,obj.first,
                           obj.head_pos,obj.hypernyms,obj.higher_hypernyms,
                           #obj.hhead,obj.hhead_pos,obj.deprel,
                           obj.prevprev,obj.prev,
                           obj.post,obj.postpost,obj.prevprev_pos,obj.prev_pos,
                           obj.post_pos,obj.postpost_pos,obj.target]
                except:
                    traceback.print_exc()
                    print('smth wrong:',chunks,'\n',subtree)
            else:
                prevs,posts,start_idxs,end_idxs = maintain_queues(subtree,prevs,posts,start_idxs,end_idxs,spans)
        else:
            prevs,posts,start_idxs,end_idxs = maintain_queues(subtree,prevs,posts,start_idxs,end_idxs,spans)
