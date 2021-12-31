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
from betacode.conv import beta_to_uni as bu
import logging

logging.basicConfig(
    filename="data/log.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


def carrega_texto(
    nome_arquivo: str, diorisis_path: str, verbose: bool = True
) -> Dict[str, List[Any]]:
    """docstring for carrega"""

    arquivo_path = diorisis_path + nome_arquivo

    if verbose:
        print(f"Carregando {nome_arquivo}")
        logging.info(f"Carregando {nome_arquivo}")

    with open(arquivo_path, "r") as arquivo:
        texto: List[Any] = json.load(arquivo)["sentences"]

    return {nome_arquivo: texto}


def carrega_autor(
    autor: str, diorisis_path: str, ignore: List[str] = [], verbose: bool = True
) -> Dict[str, List[Any]]:
    """docstring for carrega_autor"""
    textos_autor = [
        x
        for x in os.listdir(diorisis_path)
        if (x.startswith(autor) and x not in ignore)
    ]
    corpus_autor: Dict[str, List[Any]] = dict()

    for texto in textos_autor:
        corpus_autor.update(carrega_texto(texto, diorisis_path, verbose))

    return corpus_autor


def carrega_textos(
    autores: List[str], diorisis_path: str, ignore: List[str] = [], verbose: bool = True
) -> Dict[str, List[Any]]:
    """
    docstring for carrega_textos
    """
    corpus: Dict[str, List[Any]] = dict()

    for autor in autores:
        corpus.update(carrega_autor(autor, diorisis_path, ignore, verbose))

    return corpus


def em_pandas(corpus: Dict[str, List[Any]], verbose: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "sent_id",
            "location",
            "form",
            "lemma",
            "POS",
            "analyses",
            "id",
            "file",
            "author",
            "text",
        ]
    )
    for nome_arquivo in corpus.keys():

        if verbose:
            print(f"Criando DF para: {nome_arquivo}")
            logging.info(f"Criando DF para: {nome_arquivo}")

        sents = corpus[nome_arquivo]

        data: List[Dict[str, Union[str, int, float]]] = list()

        for sent in sents:
            sent_id = sent["id"]
            sent_location = sent["location"]
            for t in sent["tokens"]:
                t_data: Dict[str, Union[str, int, float]] = dict()
                t_data["sent_id"] = int(sent_id)
                t_data["location"] = sent_location
                if t["type"] == "word":
                    t_data["form"] = bu(t["form"])
                    if "entry" in t["lemma"].keys():
                        t_data["lemma"] = bu(t["lemma"]["entry"])
                    else:
                        t_data["lemma"] = np.nan
                    if "POS" in t["lemma"].keys():
                        t_data["POS"] = t["lemma"]["POS"]
                    else:
                        t_data["POS"] = np.nan
                    t_data["analyses"] = ";".join(t["lemma"]["analyses"])
                    t_data["id"] = int(t["id"])
                elif t["type"] == "punct":
                    t_data["form"] = bu(t["form"])
                    t_data["lemma"] = bu(t["form"])
                    t_data["POS"] = "punct"
                    t_data["analyses"] = "punct"
                    t_data["id"] = 0
                t_data["file"] = nome_arquivo

                data.append(t_data)

        d_df = pd.DataFrame(data)
        d_df[["author", "text"]] = d_df["file"].str.split("-", n=1, expand=True)
        d_df["text"] = (
            d_df["text"].str.replace(r"\([0-9]*\).json", "", regex=True).str.strip()
        )
        d_df["author"] = (
            d_df["author"].str.replace(r"\([0-9]*\)", "", regex=True).str.strip()
        )
        df = pd.concat([df, d_df])

    return df


def sent_pandas(corpus: Dict[str, List[Any]], verbose: bool = False):

    df = pd.DataFrame(columns=["sent_id", "location", "forms", "lemmata", "file"])
    for nome_arquivo in corpus.keys():

        if verbose:
            print(f"Criando DF para: {nome_arquivo}")
            logging.info(f"Criando DF para: {nome_arquivo}")

        sents = corpus[nome_arquivo]

        data: List[Dict[str, Union[str, int, float]]] = list()

        for sent in sents:
            s_dict = {}
            s_dict["sent_id"] = sent["id"]
            s_dict["location"] = sent["location"]
            s_dict["forms"] = bu(" ".join([x["form"] for x in sent["tokens"]]))
            s_dict["lemmata"] = bu(
                " ".join(
                    [
                        x["lemma"].get("entry", "0")
                        for x in sent["tokens"]
                        if x["type"] != "punct"
                    ]
                )
            )

            s_dict["file"] = nome_arquivo
            data.append(s_dict)

        d_df = pd.DataFrame(data)
        d_df[["author", "text"]] = d_df["file"].str.split("-", n=1, expand=True)
        d_df["text"] = (
            d_df["text"].str.replace(r"\([0-9]*\).json", "", regex=True).str.strip()
        )
        d_df["author"] = (
            d_df["author"].str.replace(r"\([0-9]*\)", "", regex=True).str.strip()
        )
        df = pd.concat([df, d_df])

    return df
