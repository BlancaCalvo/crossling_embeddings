from gensim.models import Word2Vec
import argparse
import pandas as pd
import spacy
nlp_es = spacy.load("es_core_news_md")
nlp_en = spacy.load("en_core_web_md")

parser = argparse.ArgumentParser()
parser.add_argument('--es_corpus', type=str, help='input file path for Spanish', required=False)
parser.add_argument('--en_corpus', type=str, help='input file path for English', required=False)
args = parser.parse_args()

def preprocess_es(data):
    df = pd.read_csv(data)#[:10000]
    print('Imported')
    text_es = list(df['TEXTO'])
    #text_es = ' '.join(str(text_es))
    sentences = []
    for e in text_es:
        doc = nlp_es(str(e))
        for sent in doc.sents:
            sentences.append(str(sent))

    tokenised_sentences = []
    for line in sentences:
        sent = nlp_es(line)
        token_result = []
        for token in sent:
            token_result.append(str(token))
        tokenised_sentences.append(token_result)

    return tokenised_sentences

def preprocess_en(data):
    df = pd.read_csv(data, sep='\t', names=['id', 'sentence'])
    sentences = list(df['sentence'])

    tokenised_sentences = []
    for line in sentences:
        sent = nlp_en(line)
        token_result = []
        for token in sent:
            token_result.append(str(token))
        tokenised_sentences.append(token_result)

    return tokenised_sentences

if args.es_corpus:
    es_tokenised_sentences = preprocess_es(args.es_corpus)
    print(len(es_tokenised_sentences))
    model = Word2Vec(es_tokenised_sentences)
    model.save('model_es.bin')
    model.wv.save_word2vec_format('model_es.txt', binary=False)
    print('Spanish model saved.')

if args.en_corpus:
    en_tokenised_sentences = preprocess_en(args.en_corpus)
    print(len(en_tokenised_sentences))
    model = Word2Vec(en_tokenised_sentences)
    model.save('model_en.bin')
    model.wv.save_word2vec_format('model_en.txt', binary=False)
    print('English model saved.')

#sentences = es_tokenised_sentences + en_tokenised_sentences

#model = Word2Vec(sentences)

#model.save('model.bin')

#model.wv.save_word2vec_format('model.txt', binary=False)