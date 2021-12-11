"""
File: diorisis_reader.py
Author: Caio Geraldes
Email: caio.geraldes@usp.br
Github: https://github.com/caiogeraldes
"""

import os
import json
from typing import List,Dict,Any

def carrega_texto(nome_arquivo: str, diorisis_path: str) -> Dict[str,List[Any]]:
    """docstring for carrega"""
    if diorisis_path[-1] != "/":
        diorisis_path = diorisis_path + "/"

    arquivo_path = diorisis_path + nome_arquivo

    with open(arquivo_path, 'r') as arquivo:
        texto: List[Any] = json.load(arquivo)['sentences']

    return {nome_arquivo: texto}


def carrega_autor(autor: str, diorisis_path: str) -> Dict[str, List[Any]]:
    """ docstring for carrega_autor """
    textos_autor = [x for x in os.listdir(diorisis_path) if x.startswith(autor)]
    corpus_autor: Dict[str, List[Any]] = dict()

    for texto in textos_autor:
        corpus_autor.update(carrega_texto(texto, diorisis_path))

    return corpus_autor


def carrega_autores(autores: List[str], diorisis_path: str) -> Dict[str, List[Any]]:
    corpus: Dict[str, List[Any]] = dict()

    for autor in autores:
        corpus.update(carrega_autor(autor, diorisis_path))

    return corpus
