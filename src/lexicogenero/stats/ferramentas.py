"""
File: ferramentas.py
Author: Caio Geraldes
Email: caio.geraldes@usp.br
Github: https://github.com/caiogeraldes
Description: Funções de apoio para aplicação de modelos estatísticos
"""
import itertools


def achatar(lista):
    """
    Recebe uma lista de n dimensões e retorna uma lista de n-1 dimensões.
    """
    return list(itertools.chain(*lista))
