"""Gibbs sampler for the Hierarchical LDA model.

This module implements a simple Gibbs sampler for hierarchical Latent Dirichlet
Allocation (hLDA). The algorithm follows the nested Chinese restaurant process
described in the original hLDA papers:

* Blei et al.,
  "Hierarchical Topic Models and the Nested Chinese Restaurant Process".
* Griffiths et al.,
  "The Nested Chinese Restaurant Process and Bayesian"
  " Nonparametric Inference of Topic Hierarchies".

It provides helper functions for loading data and the :class:`HierarchicalLDA`
sampler.
"""

import csv
from math import log
import sys

from numpy.random import RandomState

import numpy as np


class NCRPNode(object):

    # class variable to keep track of total nodes created so far
    total_nodes = 0
    last_node_id = 0

    def __init__(self, num_levels, vocab, parent=None, level=0,
                 random_state=None):

        self.node_id = NCRPNode.last_node_id
        NCRPNode.last_node_id += 1

        self.customers = 0
        self.parent = parent
        self.children = []
        self.level = level
        self.total_words = 0
        self.num_levels = num_levels

        self.vocab = np.array(vocab)
        self.word_counts = np.zeros(len(vocab))

        if random_state is None:
            self.random_state = RandomState()
        else:
            self.random_state = random_state

    def __repr__(self):
        parent_id = None
        if self.parent is not None:
            parent_id = self.parent.node_id
        return (
            f"Node={self.node_id} level={self.level} "
            f"customers={self.customers} "
            f"total_words={self.total_words} parent={parent_id}"
        )

    def add_child(self):
        ''' Adds a child to the next level of this node '''
        node = NCRPNode(
            self.num_levels,
            self.vocab,
            parent=self,
            level=self.level + 1,
            random_state=self.random_state,
        )
        self.children.append(node)
        NCRPNode.total_nodes += 1
        return node

    def is_leaf(self):
        ''' Check if this node is a leaf node '''
        return self.level == self.num_levels-1

    def get_new_leaf(self):
        ''' Keeps adding nodes along the path until a leaf node is generated'''
        node = self
        for lvl in range(self.level, self.num_levels - 1):
            node = node.add_child()
        return node

    def drop_path(self):
        """Remove a document from a path starting at this leaf and moving
        upwards."""
        node = self
        while node is not None:
            node.customers -= 1
            if node.customers == 0 and node.parent is not None:
                node.parent.remove(node)
            node = node.parent

    def remove(self, node):
        ''' Removes a child node '''
        self.children.remove(node)
        NCRPNode.total_nodes -= 1

    def add_path(self):
        ''' Adds a document to a path starting from this node '''
        node = self
        node.customers += 1
        for level in range(1, self.num_levels):
            node = node.parent
            node.customers += 1

    def select(self, gamma):
        """Select an existing child or create a new one using the CRP."""

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
        sorted_vocab = self.vocab[pos]
        sorted_vocab = sorted_vocab[:n_words]
        sorted_weights = self.word_counts[pos]
        sorted_weights = sorted_weights[:n_words]

        output = ''
        for word, weight in zip(sorted_vocab, sorted_weights):
            if with_weight:
                output += '%s (%d), ' % (word, weight)
            else:
                output += '%s, ' % word
        return output


class HierarchicalLDA(object):
    """Collapsed Gibbs sampler for hierarchical LDA.

    Parameters
    ----------
    corpus : Sequence[Sequence[int]]
        Collection of documents encoded as lists of token ids.
    vocab : Sequence[str]
        Mapping from token id to word.
    alpha : float, optional
        Dirichlet prior for document-topic distributions.
    gamma : float, optional
        Parameter of the nested CRP controlling branching behaviour.
    eta : float, optional
        Dirichlet prior for topic-word distributions.
    seed : int, optional
        Seed for the internal random number generator.
    verbose : bool, optional
        Whether to print progress during sampling.
    num_levels : int, optional
        Depth of the topic hierarchy.

    Attributes
    ----------
    root_node : NCRPNode
        Root of the NCRP tree representing the topic hierarchy.
    document_leaves : dict[int, NCRPNode]
        Mapping from document index to the leaf node currently assigned.
    levels : ndarray
        Array of per-word level assignments for each document.
    random_state : RandomState
        Random number generator used by the sampler.
    """

    def __init__(self, corpus, vocab,
                 alpha=10.0, gamma=1.0, eta=0.1,
                 seed=0, verbose=True, num_levels=3):

        NCRPNode.total_nodes = 0
        NCRPNode.last_node_id = 0

        self.corpus = corpus
        self.vocab = vocab
        # smoothing on doc-topic distributions
        self.alpha = alpha
        # "imaginary" customers at the next, as yet unused table
        self.gamma = gamma
        # smoothing on topic-word distributions
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
        path = np.zeros(self.num_levels, dtype=object)

        # initialize and fill the topic pointer arrays for
        # every document. Set everything to the single path that
        # we added earlier.
        self.root_node = NCRPNode(
            self.num_levels,
            self.vocab,
            random_state=self.random_state,
        )
        # currently selected path (i.e. leaf node) through the NCRP tree
        self.document_leaves = {}
        # indexed < doc, token >
        self.levels = np.zeros(self.num_documents, dtype=object)
        for d in range(len(self.corpus)):

            # populate nodes into the path of this document
            doc = self.corpus[d]
            doc_len = len(doc)
            path[0] = self.root_node
            # always add to the root node first
            self.root_node.customers += 1
            for level in range(1, self.num_levels):
                # at each level, a node is selected by its parent node based on
                # the CRP prior
                parent_node = path[level-1]
                level_node = parent_node.select(self.gamma)
                level_node.customers += 1
                path[level] = level_node

            # set the leaf node for this document
            leaf_node = path[self.num_levels-1]
            self.document_leaves[d] = leaf_node

            # randomly assign each word in the document to a level (node)
            # along the path
            self.levels[d] = np.zeros(doc_len, dtype=int)
            for n in range(doc_len):
                w = doc[n]
                random_level = self.random_state.randint(self.num_levels)
                random_node = path[random_level]
                random_node.word_counts[w] += 1
                random_node.total_words += 1
                self.levels[d][n] = random_level

    def estimate(
        self,
        num_samples,
        display_topics=50,
        n_words=5,
        with_weights=True,
    ):

        print('HierarchicalLDA sampling\n')
        for s in range(num_samples):

            sys.stdout.write('.')

            for d in range(len(self.corpus)):
                self.sample_path(d)

            for d in range(len(self.corpus)):
                self.sample_topics(d)

            if (s > 0) and ((s+1) % display_topics == 0):
                print(f" {s+1}")
                self.print_nodes(n_words, with_weights)
#                 print

    def sample_path(self, d):

        # define a path starting from the leaf node of this doc
        path = np.zeros(self.num_levels, dtype=object)
        node = self.document_leaves[d]
        for level in range(self.num_levels - 1, -1, -1):
            # e.g. [3, 2, 1, 0] for num_levels = 4
            path[level] = node
            node = node.parent

        # remove this document from the path, deleting empty nodes if necessary
        self.document_leaves[d].drop_path()

        ############################################################
        # calculates the prior p(c_d | c_{-d}) in eq. (4)
        ############################################################

        node_weights = {}
        self.calculate_ncrp_prior(node_weights, self.root_node, 0.0)

        ############################################################
        # calculates the likelihood p(w_d | c, w_{-d}, z) in eq. (4)
        ############################################################

        level_word_counts = {}
        for level in range(self.num_levels):
            level_word_counts[level] = {}
        doc_levels = self.levels[d]
        doc = self.corpus[d]

        # remove doc from path
        for n in range(len(doc)):  # for each word in the doc

            # count the word at each level
            level = doc_levels[n]
            w = doc[n]
            if w not in level_word_counts[level]:
                level_word_counts[level][w] = 1
            else:
                level_word_counts[level][w] += 1

            # remove word count from the node at that level
            level_node = path[level]
            level_node.word_counts[w] -= 1
            level_node.total_words -= 1
            assert level_node.word_counts[w] >= 0
            assert level_node.total_words >= 0

        self.calculate_doc_likelihood(node_weights, level_word_counts)

        ############################################################
        # pick a new path
        ############################################################

        nodes = np.array(list(node_weights.keys()))
        weights = np.array([node_weights[node] for node in nodes])
        # normalise so the largest weight is 1
        weights = np.exp(weights - np.max(weights))
        weights = weights / np.sum(weights)

        choice = self.random_state.multinomial(1, weights).argmax()
        node = nodes[choice]

        # if we picked an internal node, we need to add a new path to the leaf
        if not node.is_leaf():
            node = node.get_new_leaf()

        # add the doc back to the path
        node.add_path()  # add a customer to the path
        self.document_leaves[d] = node  # store the leaf node for this doc

        # add the words
        for level in range(self.num_levels - 1, -1, -1):
            # e.g. [3, 2, 1, 0] for num_levels = 4
            word_counts = level_word_counts[level]
            for w in word_counts:
                node.word_counts[w] += word_counts[w]
                node.total_words += word_counts[w]
            node = node.parent

    def calculate_ncrp_prior(self, node_weights, node, weight):
        ''' Calculates the prior on the path according to the nested CRP '''
        for child in node.children:
            child_weight = log(float(child.customers) /
                               (node.customers + self.gamma))
            self.calculate_ncrp_prior(node_weights, child,
                                      weight + child_weight)

        if node.is_leaf():
            node_weights[node] = weight
        else:
            node_weights[node] = weight + log(self.gamma /
                                              (node.customers + self.gamma))

    def calculate_doc_likelihood(self, node_weights, level_word_counts):

        # calculate the weight for a new path at a given level
        new_topic_weights = np.zeros(self.num_levels)
        for level in range(1, self.num_levels):  # skip the root

            word_counts = level_word_counts[level]
            total_tokens = 0

            for w in word_counts:
                count = word_counts[w]
                for i in range(count):  # iterate over each occurrence
                    new_topic_weights[level] += log(
                        (self.eta + i) / (self.eta_sum + total_tokens)
                    )
                    total_tokens += 1

        self.calculate_word_likelihood(
            node_weights,
            self.root_node,
            0.0,
            level_word_counts,
            new_topic_weights,
            0,
        )

    def calculate_word_likelihood(
        self,
        node_weights,
        node,
        weight,
        level_word_counts,
        new_topic_weights,
        level,
    ):

        # first calculate the likelihood of the words at this level
        # given this topic
        node_weight = 0.0
        word_counts = level_word_counts[level]
        total_words = 0

        for w in word_counts:
            count = word_counts[w]
            for i in range(count):  # iterate over each occurrence
                node_weight += log(
                    (self.eta + node.word_counts[w] + i)
                    / (self.eta_sum + node.total_words + total_words)
                )
                total_words += 1

        # propagate that weight to the child nodes
        for child in node.children:
            self.calculate_word_likelihood(
                node_weights,
                child,
                weight + node_weight,
                level_word_counts,
                new_topic_weights,
                level + 1,
            )

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
        level_counts = np.zeros(self.num_levels, dtype=int)
        for c in doc_levels:
            level_counts[c] += 1

        # get the leaf node and populate the path
        path = np.zeros(self.num_levels, dtype=object)
        node = self.document_leaves[d]
        for level in range(self.num_levels - 1, -1, -1):
            # e.g. [3, 2, 1, 0] for num_levels = 4
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
                level_weights[level] = (
                    (self.alpha + level_counts[level])
                    * (self.eta + path[level].word_counts[w])
                    / (self.eta_sum + path[level].total_words)
                )
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
        out = "    " * indent
        out += (
            f"topic={node.node_id} level={node.level} "
            f"(documents={node.customers}): "
        )
        out += node.get_top_words(n_words, with_weights)
        print(out)
        for child in node.children:
            self.print_node(child, indent+1, n_words, with_weights)

    def export_tree(self):
        """Return the current hierarchy as a JSON‑serialisable structure."""

        def visit(node):
            return {
                "id": int(node.node_id),
                "level": int(node.level),
                "customers": int(node.customers),
                "total_words": int(node.total_words),
                "children": [visit(child) for child in node.children],
            }

        return visit(self.root_node)


def load_vocab(file_name):
    with open(file_name, 'r', encoding='utf-8', newline='') as f:
        vocab = []
        reader = csv.reader(f)
        for row in reader:
            idx, word = row
            stripped = word.strip()
            vocab.append(stripped)
        return vocab


def load_corpus(file_name):
    with open(file_name, 'r', encoding='utf-8', newline='') as f:
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
