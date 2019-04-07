# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:51:35 2019

@author: aaq109
For queries Contact: awais.ashfaq@hh.se
"""
from __future__ import print_function
from gensim.models import doc2vec
from collections import namedtuple
import scipy.io as sio


label = 'sampledata_d2v.csv'
admDiagMap = {}
infd = open(label, 'r')
infd.readline()
for line in infd:
    tokens = line.strip().split(',')
    admId = (tokens[0])
    d = (tokens[1])
    diagId = d.replace('"', '')
    if admId in admDiagMap:
        admDiagMap[admId].append(diagId)
    else:
        admDiagMap[admId] = [diagId]

infd.close()

s1 = list(admDiagMap.values())

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(s1):
    words = text
    tags = [i]
    docs.append(analyzedDocument(words, tags))

Emb_size=185 #size of the visit vector K
window=121 # Max length of codes in any visit
min_count=0 # Consider all codes
ns=20 # Negative sampling
ns_exponent=-0.75 # Negative because we like to account for rare clinical events
dm=0 # for PV-DBOW


model = doc2vec.Doc2Vec(docs, size = Emb_size, window = window, min_count = min_count, workers = 4,negative =ns, ns_exponent=ns_exponent, dm=dm)

# Get the vectorsand save

d2v=model.docvecs.doctag_syn0
sio.savemat('d2v_185.mat', {'d2v_185':d2v})
