from conllu import parse_tree, print_tree


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



def find_head_verb(node):
    curr_ns = {}
    if node.data['xpostag'].startswith('V'):
        for child in node.children:
            if child.data['xpostag'].startswith('N'):
                curr_ns[(child.data['id'],
                         child.data['form'])] = (node.data['lemma'].lower(),
                                                 node.data['xpostag'],
                                                 child.data['deprel'])
    if node.data['xpostag'].startswith('N'):
        for child in node.children:
            if child.data['deprel'] == 'cop':
                curr_ns[(node.data['id'],
                         node.data['form'])] = (child.data['lemma'].lower(),
                                                child.data['xpostag'],
                                                child.data['deprel'])
    for child in node.children:
        curr_ns.update(find_head_verb(child))
    return curr_ns

def find_head(node):
    curr_ns = {}
    for child in node.children:
        if child.data['xpostag'].startswith('N'):
            curr_ns[(child.data['id'],
                     child.data['form'])] = (node.data['lemma'].lower(),
                                             node.data['xpostag'],
                                             child.data['deprel'])
    if node.data['xpostag'].startswith('N'):
        for child in node.children:
            if child.data['deprel'] == 'cop':
                curr_ns[(node.data['id'],
                         node.data['form'])] = (child.data['lemma'].lower(),
                                                child.data['xpostag'],
                                                child.data['deprel'])
    for child in node.children:
        curr_ns.update(find_head(child))
    return curr_ns
        
    
    
