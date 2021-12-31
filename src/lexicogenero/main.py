#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
    classification_report,
    log_loss,
)
from cltk.alphabet.grc import normalize_grc
from lexicogenero.ferramentas.diorisis_reader import (
    carrega_textos,
    em_pandas,
    sent_pandas,
)
from lexicogenero.ferramentas.data import gera_hist_filo, gera_paragrafo, gera_sent
from lexicogenero.grc import STOPS_LIST

import logging

logging.basicConfig(
    filename="data/log.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

# Carrega path para o Diorisis a depender do especificado em ../.env,
# rompe runtime caso não esteja especificada.
load_dotenv()
DIORISIS_PATH = os.getenv("DIORISIS_PATH")
PERSEUS_PATH = os.getenv("PERSEUS_PATH")
assert DIORISIS_PATH is not None, "Path para DIORISIS não especificada"
assert DIORISIS_PATH is not None, "Path para PERSEUS não especificada"

if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = [20, 5]
    plt.style.use("ggplot")
    sns.set_palette("Dark2")

    print("Gerando banco de dados")
    logging.info("Gerando banco de dados")

    DATA = "/home/silenus/proj/trabalho-lingcomp/data/data_classico.csv"
    SENTS = "/home/silenus/proj/trabalho-lingcomp/data/sents_classico.csv"
    if os.path.exists(DATA) and os.path.exists(SENTS):
        print(f"Carregando arquivos:\n\t{DATA}\n\t{SENTS}")
        logging.info(f"Carregando arquivos:\n\t{DATA}\n\t{SENTS}")
        df_tokens: pd.Dataframe = pd.read_csv(DATA)
        df_sents: pd.Dataframe = pd.read_csv(SENTS)
    else:
        ignorados = [
            "Xenophon (0032) - On the Art of Horsemanship (013).json",
            "Xenophon (0032) - Economics (003).json",
            "Xenophon (0032) - Ways and Means (011).json",
            "Xenophon (0032) - Constitution of the Lacedaemonians (010).json",
            "Xenophon (0032) - On the Cavalry Commander (012).json",
            "Xenophon (0032) - On Hunting (014).json",
            "Xenophon (0032) - Apology (005).json",
            "Plato (0059) - Lovers (016).json",  # Espúrios
            "Plato (0059) - Epistles (036).json",
            "Plato (0059) - Alcibiades 1 (013).json",
            "Plato (0059) - Alcibiades 2 (014).json",  # Anotação problemática
            "Plato (0059) - Cleitophon (029).json",
            "Plato (0059) - Epinomis (035).json",
            "Plato (0059) - Hipparchus (015).json",
            "Plato (0059) - Menexenus (028).json",
            "Plato (0059) - Minos (033).json",
            "Plato (0059) - Theages (017).json",
        ]

        corpus = carrega_textos(
            autores=[
                "Herodotus",
                "Thucydides",
                "Plato",
                "Xenophon (0032)",  # Exclui Xenofonte de Éfeso
            ],
            diorisis_path=DIORISIS_PATH,
            ignore=ignorados,
            verbose=False,
        )
        df_tokens = em_pandas(corpus)
        df_sents = sent_pandas(corpus)
        del corpus

        print(f"Salvando dados em tokens em {DATA}")
        logging.info(f"Salvando dados em tokens em {DATA}")
        df_tokens.to_csv(DATA, index=False)
        print(f"Salvando dados em sentenças em {SENTS}")
        logging.info(f"Salvando dados em sentenças em {SENTS}")
        df_sents.to_csv(SENTS, index=False)

    df_tokens.dropna(inplace=True)
    df_sents.dropna(inplace=True)

    print("Normalizando dados")
    logging.info("Normalizando dados")
    df_sents["lemmata"] = df_sents["lemmata"].apply(normalize_grc)
    df_tokens["lemma"] = df_tokens["lemma"].apply(normalize_grc)

    print("Anotando gênero")
    logging.info("Anotando gênero")
    lst_hist = [
        "Herodotus (0016) - Histories (001).json",
        "Xenophon (0032) - Hellenica (001).json",
        "Xenophon (0032) - Cyropaedia (007).json",
        "Xenophon (0032) - Anabasis (006).json",
        "Thucydides (0003) - History (001).json",
    ]

    df_tokens = gera_hist_filo(df_tokens, lst_hist)
    df_sents = gera_hist_filo(df_sents, lst_hist)

    print("Separando subcorpora")
    logging.info("Separando subcorpora")
    df_verbos = df_tokens.loc[
        (df_tokens.POS == "verb") & (-df_tokens.lemma.isin(STOPS_LIST)),
    ]
    df_verbos_sent = gera_sent(df_verbos)
    df_verbos_par = gera_paragrafo(df_verbos)

    df_subst = df_tokens.loc[
        (df_tokens.POS == "noun") & (-df_tokens.lemma.isin(STOPS_LIST)),
    ]
    df_subst_sent = gera_sent(df_subst)
    df_subst_par = gera_paragrafo(df_subst)

    df_tokens_par = gera_paragrafo(df_tokens.loc[df_tokens.POS != "punct"])

    print("Validando modelos")
    logging.info("Validando modelos")
    x_dsv, y_dsv = df_verbos_sent.lemma, df_verbos_sent.genero
    x_dst, y_dst = df_sents.lemmata, df_sents.genero
    x_dpt, y_dpt = df_tokens_par.lemma, df_tokens_par.genero
    x_dpv, y_dpv = df_verbos_par.lemma, df_verbos_par.genero
    x_dss, y_dss = df_subst_sent.lemma, df_subst_sent.genero
    x_dps, y_dps = df_subst_par.lemma, df_subst_par.genero

    modelos = {
        "doc-sent-verbo": (x_dsv, y_dsv),
        "doc-sent-subst": (x_dss, y_dss),
        "doc-sent-token": (x_dst, y_dst),
        "doc-par-verbo": (x_dpv, y_dpv),
        "doc-par-subst": (x_dps, y_dps),
        "doc-par-token": (x_dpt, y_dpt),
    }

    resultados = []

    cv = 10
    for modelo, (x, y) in modelos.items():
        pipe = Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(ngram_range=(1, 1), stop_words=STOPS_LIST),
                ),
                ("bayes", MultinomialNB()),
            ]
        )
        print(f"Modelo: MultinomialNB - {modelo}")
        logging.info(f"Modelo: MultinomialNB - {modelo}")
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            x, y, test_size=0.20, shuffle=True
        )
        pipe.fit(x_treino, y_treino)
        y_pred = pipe.predict(x_teste)

        scores = cross_validate(
            pipe,
            x_treino,
            y_treino,
            cv=cv,
            scoring={
                "precisão_filo": make_scorer(precision_score, pos_label="filo"),
                "cobertura_filo": make_scorer(recall_score, pos_label="filo"),
                "f1_filo": make_scorer(f1_score, pos_label="filo"),
                "acurácia": make_scorer(accuracy_score),
            },
            return_train_score=False,
        )
        scores["classe"] = [modelo for _ in range(cv)]
        scores["nb"] = ["MultinomialNB" for _ in range(cv)]
        resultados.append(scores)

    for modelo, (x, y) in modelos.items():
        pipe = Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        ngram_range=(1, 1), stop_words=STOPS_LIST, binary=True
                    ),
                ),
                ("bayes", BernoulliNB()),
            ]
        )
        print(f"Modelo: Bernoulli - {modelo}")
        logging.info(f"Modelo: Bernoulli - {modelo}")
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            x, y, test_size=0.20, shuffle=True
        )
        pipe.fit(x_treino, y_treino)
        y_pred = pipe.predict(x_teste)

        scores = cross_validate(
            pipe,
            x_treino,
            y_treino,
            cv=cv,
            scoring={
                "precisão_filo": make_scorer(precision_score, pos_label="filo"),
                "cobertura_filo": make_scorer(recall_score, pos_label="filo"),
                "f1_filo": make_scorer(f1_score, pos_label="filo"),
                "acurácia": make_scorer(accuracy_score),
            },
            return_train_score=False,
        )
        scores["classe"] = [modelo for _ in range(cv)]
        scores["nb"] = ["Bernoulli" for _ in range(cv)]
        resultados.append(scores)

    for modelo, (x, y) in modelos.items():
        pipe = Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(ngram_range=(1, 1), stop_words=STOPS_LIST),
                ),
                ("bayes", ComplementNB()),
            ]
        )
        print(f"Modelo: Complement - {modelo}")
        logging.info(f"Modelo: Complement - {modelo}")
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            x, y, test_size=0.20, shuffle=True
        )
        pipe.fit(x_treino, y_treino)
        y_pred = pipe.predict(x_teste)

        scores = cross_validate(
            pipe,
            x_treino,
            y_treino,
            cv=cv,
            scoring={
                "precisão_filo": make_scorer(precision_score, pos_label="filo"),
                "cobertura_filo": make_scorer(recall_score, pos_label="filo"),
                "f1_filo": make_scorer(f1_score, pos_label="filo"),
                "acurácia": make_scorer(accuracy_score),
            },
            return_train_score=False,
        )
        scores["classe"] = [modelo for _ in range(cv)]
        scores["nb"] = ["ComplementNB" for _ in range(cv)]
        resultados.append(scores)

    df_res = pd.DataFrame(resultados)
    df_res = df_res.explode(df_res.columns.to_list()).reset_index(drop=True)
    df_res = df_res.melt(["classe", "nb"], var_name="métrica", value_name="vals")
    df_res = df_res.loc[df_res.métrica != "fit_time"]
    df_res = df_res.loc[df_res.métrica != "score_time"]
    print(
        "Salvando resultados em /home/silenus/proj/trabalho-lingcomp/data/data_res.csv"
    )
    logging.info(
        "Salvando resultados em /home/silenus/proj/trabalho-lingcomp/data/data_res.csv"
    )
    df_res.to_csv("/home/silenus/proj/trabalho-lingcomp/data/data_res.csv", index=False)

    print("Plotando resultados dos modelos")
    logging.info("Plotando resultados dos modelos")
    sns.set(font_scale=1)
    plot = sns.displot(
        data=df_res,
        x="vals",
        col="métrica",
        row="classe",
        hue="nb",
        palette="Dark2",
        kind="kde",
    )
    plot.set_titles(template="{col_name}")
    plt.savefig(
        "/home/silenus/proj/trabalho-lingcomp/texto/figs/performance.geral.png",
        bbox_inches="tight",
    )

    plot = sns.displot(
        data=df_res.loc[df_res.nb == "MultinomialNB"],
        x="vals",
        col="métrica",
        col_wrap=2,
        hue="classe",
        palette="Dark2",
        kind="kde",
    )
    plot.set_titles(template="{col_name}")
    plt.savefig(
        "/home/silenus/proj/trabalho-lingcomp/texto/figs/performance.mnb.png",
        bbox_inches="tight",
    )

    print("Treinando modelo de uso")
    logging.info("Treinando modelo de uso")
    verbos_dgci = set(
        [
            normalize_grc(x)
            for x in [
                "ἀνίημι",
                "δέομαι",
                "δοκέω",
                "ἐγχωρέω",
                "ἐκγίγνομαι",
                "ἔνειμι",
                "ἐντέλλω",
                "ἔοικα",
                "ἔξεστι",
                "ἐπαγγέλλω",
                "ἐπαινέω",
                "ἐπιβάλλω",
                "ἐπισκήπτω",
                "ἐπιτάσσω",
                "ἐπιτέλλω",
                "ἐπαινέω",
                "ἱκετεύω",
                "καταδικάζω",
                "κηρύσσω",
                "παραγγέλλω",
                "παραιτέω",
                "παραιτέομαι",
                "παραμυθέομαι",
                "παρίημι",
                "πιστεύω",
                "πόρω",
                "πρέπει",
                "πρέπω",
                "προβάλλω",
                "προξενέω",
                "προσδέομαι",
                "προσήκει",
                "προστάσσω",
                "προσχρῄζω",
                "προτίθημι",
                "σημαίνω",
                "συγγιγνώσκω",
                "συμβαίνω",
                "συμβουλεύω",
                "συμπίπτω",
                "ὑπάρχω",
                "ὐφίημι",
                "χρῄζω",
                "χρῄζω",
                "ὑπάρχω",
                "ἐπισκήπτω",
                "παραγγέλλω",
                "ἐγχωρέω",
                "προσήκω",
                "συμβουλεύω",
                "ἐξαρκέω",
                "ἔξεστι",
                "συμφέρω",
                "ἀφίημι",
                "δίδωμι",
                "δέομαι",
                "δοκέω",
                "ἐντέλλω",
            ]
        ]
    )

    pipe = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(ngram_range=(1, 1), stop_words=STOPS_LIST, binary=True),
            ),
            ("bayes", MultinomialNB()),
        ]
    )

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x_dpv, y_dpv, test_size=0.2, shuffle=True, random_state=169
    )

    pipe.fit(x_treino, y_treino)

    vectorizer: Any = pipe["vectorizer"]
    bayes: Any = pipe["bayes"]

    y_pred = pipe.predict_proba(x_treino)

    print(classification_report(y_pred=pipe.predict(x_teste), y_true=y_teste))
    logging.info(classification_report(y_pred=pipe.predict(x_teste), y_true=y_teste))
    print(
        f"log loss: {log_loss(y_true=y_treino, y_pred=y_pred, labels=['filo', 'hist'])}"
    )
    logging.info(
        f"log loss: {log_loss(y_true=y_treino, y_pred=y_pred, labels=['filo', 'hist'])}"
    )

    plt.figure(figsize=(15, 10))
    sns.set(font_scale=2)
    sns.heatmap(
        confusion_matrix(y_pred=pipe.predict(x_teste), y_true=y_teste),
        annot=True,
        fmt="",
    )
    plt.title("Matriz de confusão")
    plt.ylabel("Real")
    plt.xlabel("Previsão")
    plt.xticks(ticks=[0.5, 1.5], labels=["Filosofia", "Historiografia"])
    plt.yticks(ticks=[0.5, 1.5], labels=["Filosofia", "Historiografia"])
    plt.savefig(
        "/home/silenus/proj/trabalho-lingcomp/texto/figs/confusao.png",
        bbox_inches="tight",
    )

    feature_log_prob_hist: Dict[str, float] = {}
    for ngram, index in vectorizer.vocabulary_.items():
        feature_log_prob_hist[ngram] = bayes.feature_log_prob_[1][index]

    flp_hist = [
        {"genero": "hist", "lemma": lemma, "logprob": feature_log_prob_hist[lemma]}
        for lemma in sorted(
            feature_log_prob_hist, key=feature_log_prob_hist.get, reverse=True # type: ignore
        )
        if lemma in verbos_dgci
    ]

    df_flp_hist = pd.DataFrame(flp_hist)

    df_flp_hist["z"] = abs(
        (df_flp_hist.logprob - pd.Series(bayes.feature_log_prob_[1]).mean())
        / pd.Series(bayes.feature_log_prob_[1]).std()
    )

    feature_log_prob_filo = {}
    for ngram, index in vectorizer.vocabulary_.items():
        feature_log_prob_filo[ngram] = bayes.feature_log_prob_[0][index]

    flp_filo = [
        {"genero": "filo", "lemma": lemma, "logprob": feature_log_prob_filo[lemma]}
        for lemma in sorted(
            feature_log_prob_filo, key=feature_log_prob_filo.get, reverse=True # type: ignore
        )
        if lemma in verbos_dgci
    ]

    df_flp_filo = pd.DataFrame(flp_filo)

    df_flp_filo["z"] = abs(
        (df_flp_filo.logprob - pd.Series(bayes.feature_log_prob_[0]).mean())
        / pd.Series(bayes.feature_log_prob_[0]).std()
    )

    df_flp: pd.Dataframe = df_flp_filo.append(df_flp_hist).reset_index()
    print(
        "Salvando resultados em /home/silenus/proj/trabalho-lingcomp/data/data_flp.csv"
    )
    logging.info(
        "Salvando resultados em /home/silenus/proj/trabalho-lingcomp/data/data_flp.csv"
    )
    df_flp.to_csv("/home/silenus/proj/trabalho-lingcomp/data/data_flp.csv", index=False)

    sns.set_palette("Dark2")
    plt.figure(figsize=(15, 25))
    bar = sns.barplot(
        data=df_flp.sort_values("lemma"),
        y="lemma",
        x="logprob",
        hue="genero",
        orient="horiz",
    )
    for i, (z, p) in enumerate(
        zip(df_flp.sort_values("lemma").z, df_flp.sort_values("lemma").logprob) # type: ignore
    ):
        bar.text(0, (i / 2) - 0.1, str(round(z, 2)), fontdict={"fontsize": 15})
    plt.title("Probabilidade em log de cada lemma")
    plt.savefig(
        "/home/silenus/proj/trabalho-lingcomp/texto/figs/logprobs.png",
        bbox_inches="tight",
    )

    df_flp_filo.set_index("lemma", inplace=True)
    df_flp_hist.set_index("lemma", inplace=True)
    diffs = (df_flp_filo.logprob - df_flp_hist.logprob).reset_index()
    diffs["z"] = (df_flp_filo.z - df_flp_hist.z).to_frame().reset_index().z
    diffs["modal"] = [
        bool(x)
        for x in [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
        ]
    ]
    print(
        "Salvando resultados em /home/silenus/proj/trabalho-lingcomp/data/data_diff.csv"
    )
    logging.info(
        "Salvando resultados em /home/silenus/proj/trabalho-lingcomp/data/data_diff.csv"
    )
    diffs.to_csv("/home/silenus/proj/trabalho-lingcomp/data/data_diff.csv", index=False)

    plt.figure(figsize=(7, 15))
    sns.set(font_scale=1.5)

    sns.barplot(
        data=diffs.sort_values("logprob"), y="lemma", x="logprob", palette="viridis"
    )
    plt.savefig(
        "/home/silenus/proj/trabalho-lingcomp/texto/figs/diff.png", bbox_inches="tight"
    )

    plt.figure(figsize=(7, 15))
    sns.set(font_scale=1.5)

    sns.barplot(
        data=diffs.sort_values("logprob"),
        y="lemma",
        x="logprob",
        hue="modal",
        dodge=False,
        palette="viridis",
    )
    plt.savefig(
        "/home/silenus/proj/trabalho-lingcomp/texto/figs/diff2.png", bbox_inches="tight"
    )

    df_mestrado = pd.read_csv(
        "/home/silenus/docs/Academia/Mestrado/master-data/data.csv"
    )

    sns.set(font_scale=2)
    plt.figure(figsize=(7.2, 7))
    sns.countplot(
        data=df_mestrado, x="attraction", hue="dialect_attic", palette="Dark2"
    )
    plt.xlabel("Atração")
    plt.ylabel("Frequência")
    plt.legend(["Jônico", "Ático"])
    plt.savefig(fname="/home/silenus/proj/trabalho-lingcomp/texto/figs/dialeto.png")

    plt.figure(figsize=(7.2, 7))
    sns.countplot(
        data=df_mestrado,
        x="attraction",
        hue="author",
        hue_order=["Herodotus", "Xenophon", "Plato"],
        palette="Dark2",
    )
    plt.xlabel("Atração")
    plt.ylabel("Frequência")
    plt.legend(["Heródoto", "Xenofonte", "Platão"])
    plt.savefig(fname="/home/silenus/proj/trabalho-lingcomp/texto/figs/autor.png")

    plt.figure(figsize=(7.2, 7))
    sns.countplot(data=df_mestrado, x="attraction", hue="poss_verb", palette="Dark2")
    plt.xlabel("Atração")
    plt.ylabel("Frequência")
    plt.legend(["Não-modal", "Modal"])
    plt.savefig(fname="/home/silenus/proj/trabalho-lingcomp/texto/figs/vposs.png")

    plt.figure(figsize=(7.2, 7))
    sns.countplot(
        data=df_mestrado,
        x="poss_verb",
        hue="author",
        hue_order=["Herodotus", "Xenophon", "Plato"],
        palette="Dark2",
    )
    plt.xlabel("Verbo modal")
    plt.ylabel("Frequência")
    plt.legend(["Heródoto", "Xenofonte", "Platão"])
    plt.savefig(fname="/home/silenus/proj/trabalho-lingcomp/texto/figs/vposs_autor.png")

    print(
        df_flp[["genero", "lemma", "logprob", "z"]]
        .groupby(["lemma", "genero"])
        .agg(lambda x: x)
    )
    logging.info(
        df_flp[["genero", "lemma", "logprob", "z"]]
        .groupby(["lemma", "genero"])
        .agg(lambda x: x)
    )
    sns.displot(diffs.z.apply(abs), kde=True, aspect=1.5, palette="Dark2", bins=10)
    plt.yticks(ticks=[0, 2, 4, 6, 8, 10, 12, 14])
    plt.xlabel("Diferença absoluta de z")
    plt.ylabel("Conta")
    plt.axvline(diffs.z.apply(abs).quantile(0.95), color="black")
    plt.savefig(
        "/home/silenus/proj/trabalho-lingcomp/texto/figs/diffz.png", bbox_inches="tight"
    )
