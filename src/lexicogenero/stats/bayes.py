"""
File: bayes.py
Author: Caio Geraldes
Email: caio.geraldes@usp.br
Github: https://github.com/caiogeraldes
Description: Funções de apoio para aplicação do modelo Naive Bayes
"""
import numpy as np
from typing import Tuple, List, Dict, Iterable
from collections import Counter
from sklearn.model_selection import train_test_split
from lexicogenero.stats.ferramentas import achatar


def naive_bayes(x: Iterable[str],
                y: Iterable[str],
                classes: Tuple[str, str] = ('filo', 'hist'),
                random_state: int = 42):

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.20, random_state=random_state)

    classe1: List[str] = [x for x, y in zip(x_treino, y_treino) if y == classes[0]]
    classe2: List[str] = [x for x, y in zip(x_treino, y_treino) if y == classes[1]]

    vocab_classe1: Counter[str] = Counter(achatar(classe1))
    vocab_classe2: Counter[str] = Counter(achatar(classe2))

    n_classe1 = len(classe1)
    n_classe2 = len(classe2)
    n = n_classe1 + n_classe2
    p_classe1 = np.log2(n_classe1 / n)
    p_classe2 = np.log2(n_classe2 / n)

    n_vocab_classe1 = len(vocab_classe1)
    n_vocab_classe2 = len(vocab_classe2)

    def pred_bayes(texto):
        texto_classe1 = [x for x in texto if x in vocab_classe1.keys()]
        c_doc_classe1: Counter[str] = Counter(achatar(texto_classe1))
        texto_classe2 = [x for x in texto if x in vocab_classe2.keys()]
        c_doc_classe2: Counter[str] = Counter(achatar(texto_classe2))

        pf_classe1 = sum([np.log2((c_doc_classe1[token] + 1) / n_vocab_classe1 + len(texto_classe1)) for token in texto_classe1])
        pf_classe2 = sum([np.log2((c_doc_classe2[token] + 1) / n_vocab_classe2 + len(texto_classe2)) for token in texto_classe2])

        prob_classe1 = p_classe1 + pf_classe1
        prob_classe2 = p_classe2 + pf_classe2

        return prob_classe1, prob_classe2

    teste_pred_labels = []
    for t in x_teste:
        prob_classe1, prob_classe2 = pred_bayes(t)
        if prob_classe1 >= prob_classe2:
            teste_pred_labels.append(classes[0])
        else:
            teste_pred_labels.append(classes[1])

    teste_orig_labels = [x for x in y_teste]

    return teste_pred_labels, teste_orig_labels


def performance(teste_pred_labels: List[str],
                teste_orig_labels: List[str],
                classes: Tuple[str, str] = ('filo', 'hist')) -> Counter[str]:

    performance_labels = []

    for pred, label in zip(teste_pred_labels, teste_orig_labels):
        if label == classes[0] and pred == classes[0]:
            performance_labels.append("VP")
        elif label == classes[0] and pred == classes[1]:
            performance_labels.append("FP")
        elif label == classes[1] and pred == classes[1]:
            performance_labels.append("VN")
        else:
            performance_labels.append("FN")

    return Counter(performance_labels)


def metricas(perf_counter: Counter[str], verbose: bool = False) -> Dict[str, float]:

    if (perf_counter['VP'] + perf_counter['FP'] == 0) or (perf_counter['VP'] + perf_counter['FN'] == 0):
        return {'precisão': np.nan, 'cobertura': np.nan, 'acurácia': np.nan, 'medida_f1': np.nan}

    else:
        precisao = perf_counter['VP'] / (perf_counter['VP'] + perf_counter['FP'])
        cobertura = perf_counter['VP'] / (perf_counter['VP'] + perf_counter['FN'])
        acuracia = (perf_counter['VP'] + perf_counter['VN']) / sum(perf_counter.values())
        medida_f = 2 * (precisao * cobertura) / (precisao + cobertura)

        if verbose:
            print(f'Precisão:  {precisao}')
            print(f'Cobertura: {cobertura}')
            print(f'Acurácia:  {acuracia}')
            print(f'Medida F:  {medida_f}')

        return {'precisão': precisao, 'cobertura': cobertura, 'acurácia': acuracia, 'medida_f1': medida_f}
