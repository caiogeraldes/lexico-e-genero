from lexicogenero.ferramentas import diorisis_reader

DIORISIS_PATH = '/home/silenus/docs/Textos/DBs/diorisis-resources/json/'

def testa_autores():
    corpus = diorisis_reader.carrega_textos(['Herodotus'], DIORISIS_PATH)
    assert len(corpus.keys()) == 1
    corpus = diorisis_reader.carrega_textos(['Herodotus',
                                             'Appian'],
                                            DIORISIS_PATH)
    assert len(corpus.keys()) == 15

def testa_autor():
    corpus = diorisis_reader.carrega_autor('Herodotus',
                                           DIORISIS_PATH,
                                           verbose=True)
    assert len(corpus.keys()) == 1
    corpus = diorisis_reader.carrega_autor('Appian',
                                           DIORISIS_PATH)
    assert len(corpus.keys()) == 14

def testa_pandas():
    corpus = diorisis_reader.carrega_textos(['Herodotus'],
                                            DIORISIS_PATH)
    df = diorisis_reader.em_pandas(corpus, verbose=True)
    assert df.text.count() == 207940
    assert len(df.text.unique()) == 1

    corpus = diorisis_reader.carrega_textos(['Herodotus',
                                             'Appian'],
                                            DIORISIS_PATH,
                                            verbose=True)
    df = diorisis_reader.em_pandas(corpus, verbose=True)
    assert df.text.count() > 207940
    assert len(df.text.unique()) == 15
