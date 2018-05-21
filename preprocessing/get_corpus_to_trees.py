from .conllu import parse_tree, print_tree


##ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
##FORM: Word form or punctuation symbol.
##LEMMA: Lemma or stem of word form.
##UPOS: Universal part-of-speech tag.
##XPOS: Language-specific part-of-speech tag; underscore if not available.
##FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
##HEAD: Head of the current word, which is either a value of ID or zero (0).
##DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
##DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
##MISC: Any other annotation.

# target_pos is a func which returns bool from pos tag - whether it is the target tag

def find_head_verb(node,target_pos):
    curr_ns = {}
    if node.data['xpostag'].startswith('V'):
        for child in node.children:
            if target_pos(child.data['xpostag']):
                curr_ns[(child.data['id'],
                         child.data['form'])] = (node.data['lemma'].lower(),
                                                 node.data['xpostag'],
                                                 child.data['deprel'])
    if target_pos(node.data['xpostag']):
        for child in node.children:
            if child.data['deprel'] == 'cop':
                curr_ns[(node.data['id'],
                         node.data['form'])] = (child.data['lemma'].lower(),
                                                child.data['xpostag'],
                                                child.data['deprel'])
    for child in node.children:
        curr_ns.update(find_head_verb(child,target_pos))
    return curr_ns

def find_head(node,target_pos):
    curr_ns = {}
    for child in node.children:
        if target_pos(child.data['xpostag']):
            curr_ns[(int(child.data['misc']['TokenRange'].split(':')[0]),
                     child.data['form'])] = (node.data['lemma'].lower(),
                                             node.data['xpostag'],
                                             child.data['deprel'])
    if target_pos(node.data['xpostag']):
        for child in node.children:
            if child.data['deprel'] == 'cop':
                curr_ns[(int(node.data['misc']['TokenRange'].split(':')[0]),
                         node.data['form'])] = (child.data['lemma'].lower(),
                                                child.data['xpostag'],
                                                child.data['deprel'])
    for child in node.children:
        curr_ns.update(find_head(child,target_pos))
    return curr_ns

if __name__ == '__main__':
    with open('../BNC to plain text/BNC_B_10000_parsed.txt','r',encoding='utf-8') as f:
        trees = parse_tree(f.read())
    print(trees[5])
    print(find_head(trees[5],lambda x: x.startswith('N')))
    
    
