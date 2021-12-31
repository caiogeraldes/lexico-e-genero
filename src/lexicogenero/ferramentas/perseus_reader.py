import json
from lexicogenero.grc import GRC_PUNCT
from bs4 import BeautifulSoup
from cltk.alphabet.grc.beta_to_unicode import BetaCodeReplacer
from typing import List, Any

bu = BetaCodeReplacer().replace_beta_code


def lista_parágrafos(
    arquivo: str, perseus_path: str, rm_punct: bool = False, json_file: bool = False
) -> List[str]:
    """
    Retorna a lista de strings de {arquivo} do corpus Perseus localizado em
    {perseus_path} em formato unicode normalizado para uso com modelos cltk,
    stanza etc.
    A detecção de formato do arquivo (entre .json e .xml) é automática, mas
    pode ser escolhida para garantir a interpretação como json utilizando o
    parâmetro {json_file}

    Parameters:
        arquivo: str: String com o caminho para o arquivo desejado.
        perseus_path: str: String com o caminho para o corpus Perseus,
                           obtido a partir do download com a ferramenta
                           de corpora do CLTK.
        rm_punct: bool: Remove a pontuação. Default: False
        json_file: bool: Monta os dados a partir de {arquivo} em formato
                         json em oposição a xml. Default: False
    Returns:
        texto_limpo: List[str]
    """

    if arquivo.endswith(".json"):
        json_file = True

    if json_file:
        with open(f"{perseus_path}/{arquivo}", "r") as file:
            doc = json.load(file)
        texto_unicode = [
            bu(x["p"]["#text"]) for x in doc["TEI.2"]["text"]["body"]["div1"]
        ]
        texto_unicode = [x.replace("δῐ", "δι’") for x in texto_unicode]
        texto_limpo = [limpa(x, rm_punct) for x in texto_unicode if len(x) > 0]
    else:
        with open(f"{perseus_path}/{arquivo}", "r") as file:
            soup: Any = BeautifulSoup(file.read(), "lxml")

        texto_cru = soup.body.find("text").getText()
        texto_unicode = bu(texto_cru)

        texto_unicode = texto_unicode.replace(
            "δῐ", "δι’"
        )  # evita um erro irritante no cltk.
        texto_limpo = [
            limpa(x, rm_punct) for x in texto_unicode.split("\n\n") if len(x) > 0
        ]
    return texto_limpo


def limpa(string: str, rm_punct: bool = False) -> str:
    out = string.replace("\n", " ")
    if rm_punct:
        out = "".join([x for x in list(out) if x not in GRC_PUNCT])
    return out
