#!/usr/bin/python3

import os
from dotenv import load_dotenv

# Carrega path para o Diorisis a depender do especificado em ../.env,
# rompe runtime caso não esteja especificada.
load_dotenv()
DIORISIS_PATH = os.getenv("DIORISIS_PATH")
assert DIORISIS_PATH is not None, "Path para DIORISIS não especificada"


if __name__ == "__main__":
    pass
