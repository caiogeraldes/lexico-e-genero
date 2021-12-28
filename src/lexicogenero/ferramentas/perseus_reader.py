from lexicogenero.grc import GRC_PUNCT
from bs4 import BeautifulSoup
from cltk.alphabet.grc.beta_to_unicode import BetaCodeReplacer
from typing import List, Any

bu = BetaCodeReplacer().replace_beta_code


def lista_parágrafos(arquivo: str, perseus_path: str, rm_punct: bool = False) -> List[str]:
    with open(f'{perseus_path}/{arquivo}', 'r') as file:
        soup: Any = BeautifulSoup(file.read(), 'lxml')

    texto_cru = soup.body.find('text').getText()

    texto_unicode = bu(texto_cru)
    texto_unicode = texto_unicode.replace('δῐ', 'δι’') # evita um erro irritante no cltk.
    texto_limpo = [limpa(x, rm_punct) for x in texto_unicode.split('\n\n') if len(x) > 0]
    return texto_limpo


def limpa(string: str, rm_punct: bool = False) -> str:
    out = string.replace('\n', '')
    if rm_punct:
        out = "".join([x for x in list(out) if x not in GRC_PUNCT])
    return out
