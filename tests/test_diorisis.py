import os
from lexicogenero.ferramentas import diorisis_reader

DIORISIS_PATH = '/home/silenus/docs/Textos/DBs/diorisis-resources/json/'

def testa_autores():
    corpus = diorisis_reader.carrega_autores(['Herodotus'], DIORISIS_PATH)
    assert len(corpus.keys()) == 1
