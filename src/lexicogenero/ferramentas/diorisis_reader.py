"""
File: diorisis_reader.py
Author: Caio Geraldes
Email: caio.geraldes@usp.br
Github: https://github.com/caiogeraldes
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from betacode.conv import beta_to_uni


def carrega_texto(nome_arquivo: str, diorisis_path: str) -> Dict[str,
                                                                 List[Any]]:
    """docstring for carrega"""
    if diorisis_path[-1] != "/":
        diorisis_path = diorisis_path + "/"

    arquivo_path = diorisis_path + nome_arquivo

    with open(arquivo_path, 'r') as arquivo:
        texto: List[Any] = json.load(arquivo)['sentences']

    return {nome_arquivo: texto}


def carrega_autor(autor: str, diorisis_path: str) -> Dict[str, List[Any]]:
    """ docstring for carrega_autor """
    textos_autor = [x for x in os.listdir(diorisis_path)
                    if x.startswith(autor)]
    corpus_autor: Dict[str, List[Any]] = dict()

    for texto in textos_autor:
        corpus_autor.update(carrega_texto(texto, diorisis_path))

    return corpus_autor


def carrega_autores(autores: List[str], diorisis_path: str) -> Dict[str,
                                                                    List[Any]]:
    corpus: Dict[str, List[Any]] = dict()

    for autor in autores:
        corpus.update(carrega_autor(autor, diorisis_path))

    return corpus


def em_plain_text(corpus: Dict[str, List[Any]], nome_arquivo: str) -> str:
    sents = corpus[nome_arquivo]
    plain_text: str = ""

    for sent in sents:
        for token in sent['tokens']:
            plain_text += beta_to_uni(token['form']) + " "
        plain_text += "\n"
    return plain_text


def em_pandas(corpus: Dict[str, List[Any]], nome_arquivo: str) -> pd.DataFrame:
    sents = corpus[nome_arquivo]

    data: List[Dict[str, Union[str, int, float]]] = list()

    for sent in sents:
        sent_id = sent['id']
        sent_location = sent['location']
        for token in sent['tokens']:
            token_data: Dict[str, Union[str, int, float]] = dict()
            token_data['sent_id'] = int(sent_id)
            token_data['location'] = sent_location
            if token['type'] == 'word':
                token_data['form'] = beta_to_uni(token['form'])
                if 'entry' in token['lemma'].keys():
                    token_data['lemma'] = beta_to_uni(token['lemma']['entry'])
                else:
                    token_data['lemma'] = np.nan
                if 'POS' in token['lemma'].keys():
                    token_data['POS'] = token['lemma']['POS']
                else:
                    token_data['POS'] = np.nan
                token_data['analyses'] = ";".join(token['lemma']['analyses'])
                token_data['id'] = int(token['id'])
            elif token['type'] == 'punct':
                token_data['form'] = beta_to_uni(token['form'])
                token_data['lemma'] = beta_to_uni(token['form'])
                token_data['POS'] = 'punct'
                token_data['analyses'] = 'punct'
                token_data['id'] = 0
            token_data['file'] = nome_arquivo

            data.append(token_data)

    df = pd.DataFrame(data)
    df[['author', 'text']] = df['file'].str.split('-', 0, expand=True)
    df['text'] = df['text'].str.replace(r'\([0-9]*\).json', '', regex=True).str.strip()
    df['author'] = df['author'].str.replace(r'\([0-9]*\)', '', regex=True).str.strip()
    return df
