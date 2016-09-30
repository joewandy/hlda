import csv
from math import log
import sys

from numpy.random import RandomState

import numpy as np


class NCRPNode(object):
    
    # class variable to keep track of total nodes created so far
    total_nodes = 0
    last_node_id = 0
    
    def __init__(self, num_levels, vocab_list, parent=None, level=0, 
                 random_state=None):

        self.node_id = NCRPNode.last_node_id
        NCRPNode.last_node_id += 1
        
        self.customers = 0
        self.parent = parent
        self.children = []
        self.level = level        
        self.total_words = 0
        self.num_levels = num_levels

        self.vocab_list = np.array(vocab_list)
        self.word_counts = np.zeros(len(vocab_list))
                
        if random_state is None:
            self.random_state = RandomState()
        else:
            self.random_state = random_state  
            
    def __repr__(self):
        return 'Node=%d level=%d customers=%d total_words=%d parent={%s}' % (self.node_id, 
            self.level, self.customers, self.total_words, self.parent)    
                        
    def add_child(self):
        ''' Adds a child to the next level of this node '''
        node = NCRPNode(self.num_levels, self.vocab_list, parent=self, level=self.level+1)
        self.children.append(node)
        NCRPNode.total_nodes += 1
        return node

    def is_leaf(self):
        ''' Check if this node is a leaf node '''
        return self.level == self.num_levels-1
    
    def get_new_leaf(self):
        ''' Keeps adding nodes along the path until a leaf node is generated'''
        node = self
        for l in range(self.level, self.num_levels-1):
            node = node.add_child()
        return node
            
    def drop_path(self):
        ''' Removes a document from a path starting from this node '''
        node = self
        node.customers -= 1
        if node.customers == 0:
            node.parent.remove(node)
        for level in range(1, self.num_levels): # skip the root
            node = node.parent
            node.customers -= 1
            if node.customers == 0:
                node.parent.remove(node)

    def remove(self, node):
        ''' Removes a child node '''
        self.children.remove(node)
        
    def add_path(self):
        ''' Adds a document to a path starting from this node '''
        node = self
        node.customers += 1
        for level in range(1, self.num_levels):
            node = node.parent
            node.customers += 1

    def select_existing(self, gamma):
        ''' Selects an existing child by sampling according to the CRP '''
        
        weights = np.zeros(len(self.children))
        i = 0
        for child in self.children:
            weights[i] = float(child.customers) / (gamma + self.customers)
            i += 1        
        choice = self.random_state.multinomial(1, weights).argmax()
        return self.children[choice]

    def select(self, gamma):
        ''' Selects an existing child or create a new one according to the CRP '''
        
        weights = np.zeros(len(self.children)+1)
        weights[0] = float(gamma) / (gamma+self.customers)
        i = 1
        for child in self.children:
            weights[i] = float(child.customers) / (gamma + self.customers)
            i += 1

        choice = self.random_state.multinomial(1, weights).argmax()
        if choice == 0:
            return self.add_child()
        else:
            return self.children[choice-1]   
                
    def get_top_words(self, n_words, with_weight):  
        ''' Get the top n words in this node '''   

        pos = np.argsort(self.word_counts)[::-1]
        sorted_vocab_list = self.vocab_list[pos]
        sorted_vocab_list = sorted_vocab_list[:n_words]
        sorted_weights = self.word_counts[pos]
        sorted_weights = sorted_weights[:n_words]
        
        output = ''
        for word, weight in zip(sorted_vocab_list, sorted_weights):
            if with_weight:
                output += '%s (%d), ' % (word, weight)
            else:
                output += '%s, ' % word                
        return output            

class HierarchicalLDA(object):
    
    def __init__(self, corpus, vocab, vocab_list, 
                 alpha=10.0, gamma=1.0, eta=0.1, 
                 seed=0, verbose=True, num_levels=3):
        
        self.corpus = corpus
        self.vocab = vocab
        self.vocab_list = vocab_list
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta

        self.seed = seed
        self.random_state = RandomState(seed)        
        self.verbose = verbose

        self.num_levels = num_levels
        self.num_documents = len(corpus)
        self.num_types = len(vocab)
        self.eta_sum = eta * self.num_types

        # if self.verbose:        
        #     for d in range(len(self.corpus)):
        #         doc = self.corpus[d]
        #         words = ' '.join([self.vocab[n] for n in doc])
        #         print 'doc_%d = %s' % (d, words)  
        
        # initialise a single path
        path = np.zeros(self.num_levels, dtype=np.object)
        
        # initialize and fill the topic pointer arrays for 
        # every document. Set everything to the single path that 
        # we added earlier.
        self.root_node = NCRPNode(self.num_levels, self.vocab_list)
        self.document_leaves = {}
        self.levels = np.zeros(self.num_documents, dtype=np.object)
        for d in range(len(self.corpus)):
            
            # populate nodes into the path of this document
            doc = self.corpus[d]
            doc_len = len(doc)
            path[0] = self.root_node
            self.root_node.customers += 1 # always add to the root node first
            for level in range(1, self.num_levels):
                # at each level, a node is selected by its parent node based on the CRP prior
                parent_node = path[level-1]
                level_node = parent_node.select(self.gamma)
                level_node.customers += 1
                path[level] = level_node
                
            # set the leaf node for this document                 
            leaf_node = path[self.num_levels-1]
            self.document_leaves[d] = leaf_node
                        
            # randomly assign each word in the document to a level (node) along the path
            self.levels[d] = np.zeros(doc_len, dtype=np.int)
            for n in range(doc_len):
                w = doc[n]                
                random_level = self.random_state.randint(self.num_levels)
                random_node = path[random_level]
                random_node.word_counts[w] += 1
                random_node.total_words += 1
                self.levels[d][n] = random_level                

    def estimate(self, num_samples, display_topics=50, n_words=5, with_weights=True):
        
        print 'HierarchicalLDA sampling'
        for s in range(num_samples):
            
            sys.stdout.write('.')
            
            for d in range(len(self.corpus)):
                self.sample_path(d)
            
            for d in range(len(self.corpus)):
                self.sample_topics(d)
                
            if (s > 0) and (s % display_topics == 0):
                print
                self.print_nodes(n_words, with_weights)

    def sample_path(self, d):
        
        # populate path with nodes, starting from the document's leaf
        path = np.zeros(self.num_levels, dtype=np.object)
        node = self.document_leaves[d]
        for level in range(self.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            path[level] = node
            node = node.parent
            
        # remove this document from the path, deleting empty nodes if necessary
        self.document_leaves[d].drop_path()
        
        ############################################################
        # calculates p(c_d | c_{-d}) in eq. (4) of the paper
        ############################################################

        node_weights = {}
        self.calculate_ncrp(node_weights, self.root_node, 0.0)
        
        ############################################################
        # calculates p(w_d | c, w_{-d}, z) in eq. (4) of the paper
        ############################################################

        level_word_counts = {}
        for level in range(self.num_levels):
            level_word_counts[level] = {}        
        doc_levels = self.levels[d]
        doc = self.corpus[d]
        
        # save the counts of every word at each level, and remove 
        # its count from the current path
        for n in range(len(doc)):
            
            # count the word at each level
            level = doc_levels[n]
            w = doc[n]
            if w not in level_word_counts[level]:
                level_word_counts[level][w] = 1
            else:
                level_word_counts[level][w] += 1
            
            # for every word, remove its count from the node  
            # the word is assigned to (at a particular level)
            level_node = path[level]
            level_node.word_counts[w] -= 1
            level_node.total_words -= 1
            assert level_node.word_counts[w] >= 0
            assert level_node.total_words >= 0
        
        # calculate the weight for a new path at a given level
        new_topic_weights = np.zeros(self.num_levels)
        for level in range(1, self.num_levels): # skip the root
            
            word_counts = level_word_counts[level]
            total_tokens = 0
            
            for w in word_counts:
                count = word_counts[w]
                for i in range(count): # why ?????????
                    new_topic_weights[level] += log( (self.eta+i)/(self.eta_sum+total_tokens) ) 
                    total_tokens += 1
                    
        self.calculate_word_likelihood(node_weights, self.root_node, 0.0, level_word_counts, new_topic_weights, 0)            
        
        nodes = np.array(list(node_weights.keys()))
        weights = np.array([node_weights[node] for node in nodes])

        # to avoid underflow, we're using log weights and normalising the node weights
        # so that the largest weight is always 1        
        weights = np.exp(weights - np.max(weights))

        # select a node
        weights = weights / np.sum(weights)
        choice = self.random_state.multinomial(1, weights).argmax()
        node = nodes[choice]
        
        # if we have picked an internal node, we need to add a new path
        if not node.is_leaf():
            node = node.get_new_leaf()

        node.add_path()                     # add a customer to the path
        self.document_leaves[d] = node      # store the leaf node for this doc

        # update the counts of words from the document at the right level
        for level in range(self.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            word_counts = level_word_counts[level]
            for w in word_counts:
                node.word_counts[w] += word_counts[w]
                node.total_words += word_counts[w]
            node = node.parent        
        
    def calculate_ncrp(self, node_weights, node, weight):
        ''' Calculates the prior on the path according to the nested CRP '''

        for child in node.children:
            child_weight = log( float(child.customers) / (node.customers + self.gamma) )
            self.calculate_ncrp(node_weights, child, weight + child_weight)
        
        node_weights[node] = weight + log( self.gamma / (node.customers + self.gamma))
            
    def calculate_word_likelihood(self, node_weights, node, weight, level_word_counts, new_topic_weights, level):
                
        # first calculate the likelihood of the words at this level, given this topic
        node_weight = 0.0
        word_counts = level_word_counts[level]
        total_words = 0
        
        for w in word_counts:
            count = word_counts[w]
            for i in range(count): # why ?????????
                node_weight += log( (self.eta + node.word_counts[w] + i) / 
                                    (self.eta_sum + node.total_words + total_words) )
                total_words += 1
                
        # propagate that weight to the child nodes
        for child in node.children:
            self.calculate_word_likelihood(node_weights, child, weight + node_weight, 
                                           level_word_counts, new_topic_weights, level+1)
            
        # finally if this is an internal node, add the weight of a new path
        level += 1
        while level < self.num_levels:
            node_weight += new_topic_weights[level]
            level += 1
            
        node_weights[node] += node_weight
        
    def sample_topics(self, d):

        doc = self.corpus[d]
        
        # initialise level counts
        doc_levels = self.levels[d]
        level_counts = np.zeros(self.num_levels, dtype=np.int)
        for c in doc_levels:
            level_counts[c] += 1

        # get the leaf node and populate the path
        path = np.zeros(self.num_levels, dtype=np.object)
        node = self.document_leaves[d]
        for level in range(self.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            path[level] = node
            node = node.parent

        # sample a new level for each word
        level_weights = np.zeros(self.num_levels)            
        for n in range(len(doc)):

            w = doc[n]            
            word_level = doc_levels[n]

            # remove from model
            level_counts[word_level] -= 1
            node = path[word_level]
            node.word_counts[w] -= 1
            node.total_words -= 1

            # pick new level
            for level in range(self.num_levels):
                level_weights[level] = (self.alpha + level_counts[level]) *                     \
                    (self.eta + path[level].word_counts[w]) /                                   \
                    (self.eta_sum + path[level].total_words)
            level_weights = level_weights / np.sum(level_weights)
            level = self.random_state.multinomial(1, level_weights).argmax()
            
            # put the word back into the model
            doc_levels[n] = level
            level_counts[level] += 1
            node = path[level]
            node.word_counts[w] += 1
            node.total_words += 1
        
    def print_nodes(self, n_words, with_weights):
        self.print_node(self.root_node, 0, n_words, with_weights)
        
    def print_node(self, node, indent, n_words, with_weights):
        out = '    ' * indent
        out += 'topic %d (level=%d, total_words=%d, documents=%d):\t\t' % (node.node_id, node.level, node.total_words, node.customers)
        out += node.get_top_words(n_words, with_weights)
        print out        
        for child in node.children:
            self.print_node(child, indent+1, n_words, with_weights)        

def load_vocab(file_name):
    with open(file_name, 'rb') as f:
        vocab = {}
        vocab_list = []
        reader = csv.reader(f)
        for row in reader:
            idx, word = row
            stripped = word.strip()
            vocab[int(idx)] = stripped
            vocab_list.append(stripped)
        return vocab, vocab_list

def load_corpus(file_name):
    with open(file_name, 'rb') as f:
        corpus = []    
        reader = csv.reader(f)
        for row in reader:
            doc = []
            for idx_and_word in row:
                stripped = idx_and_word.strip()
                tokens = stripped.split(' ')
                if len(tokens) == 2:
                    idx, word = tokens
                    doc.append(int(idx))
            corpus.append(doc)
        return corpus
        
def main():

    vocab, vocab_list = load_vocab('/Users/joewandy/Dropbox/Meta_clustering/MS2LDA/hLDA/bbc_business_corpus.mallet.vocab.txt')
    corpus = load_corpus('/Users/joewandy/Dropbox/Meta_clustering/MS2LDA/hLDA/bbc_business_corpus.mallet.docs.txt')
    hlda = HierarchicalLDA(corpus, vocab, vocab_list)
    
    n_samples = 1000
    hlda.estimate(n_samples, display_topics=50)

if __name__ == "__main__": main()