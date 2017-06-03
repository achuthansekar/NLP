#!/usr/bin/env python3
# coding: utf-8
"""
Created on Sat Apr 15 12:45:39 2017

"""

import numpy as np
import os
import random

class StanfordSentiment:
    def __init__(self, path=None, tablesize = 1000000):
        if not path:
            path = "cs224d/datasets/stanfordSentimentTreebank"

        self.path = path
        self.tablesize = tablesize

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens
    
    def sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path + "/datasetSentences.txt", "r") as f:
            first = True
            for line in f:
                #print(line)
                if first:
                    first = False
                    continue

                splitted = line.strip().split()[1:]
                print(splitted)
                # Deal with some peculiar encoding issues with this file
                sentences += [[w.lower() for w in splitted]]
                #print(sentences)              
        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        return self._sentences
        
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

##78	At heart the movie is a deftly wrought suspense yarn whose richer shadings work as colouring rather than substance.
