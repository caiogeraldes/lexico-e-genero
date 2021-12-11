#!/usr/bin/python3

import os
from dotenv import load_dotenv

# Carrega path para o Diorisis a depender do especificado em ../.env,
# rompe runtime caso não esteja especificada.
load_dotenv()
DIORISIS_PATH = os.getenv("DIORISIS_PATH")
assert DIORISIS_PATH is not None, "Path para DIORISIS não especificada"


if __name__ == "__main__":
    from ferramentas.diorisis_reader import carrega_autores, em_plain_text, em_pandas
    print(DIORISIS_PATH)
    corpus = carrega_autores(["Herodotus"], DIORISIS_PATH)
    HDT = 'Herodotus (0016) - Histories (001).json'
    df = em_pandas(corpus, HDT)

    print(df)
