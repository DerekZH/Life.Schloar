import pandas as pd
import pickle
import gensim
import sys
import numpy as np
import nltk
from helpers.helpers import *

def lda_w2v_loader():
    ######  LDA Section
    ######  LDA Data Loading
    stop_words = nltk.corpus.stopwords.words('english')

    def tokenize(text, stop_words):
        return [token for token in gensim.utils.simple_preprocess(text) if token not in stop_words]

    class MyCorpus(gensim.corpora.TextCorpus): 
        def get_texts(self): 
            for string in self.input.values(): # for each relevant file 
                yield tokenize(string, stop_words)

    with open('Combined_Udemy_Coursera_Reviews.pickle', 'rb') as f:
        Udemy_Coursera_combined_train = pickle.load(f)

    mycorpus = MyCorpus(Udemy_Coursera_combined_train)
    mycorpus.dictionary.filter_extremes()
    sys.stdout.flush()
    mycorpus.dictionary.items()
    mycorpus_dict = gensim.corpora.dictionary.Dictionary()
    mycorpus_dict= mycorpus_dict.load('LDA_Udemy_Coursera_20_topics_gensim_dict.dict')

    lda = gensim.models.ldamulticore.LdaMulticore(id2word=mycorpus.dictionary.id2token)
    lda = lda.load('lda_full_20_topics.lda')

    LDA_results = pd.read_csv('Udemy_Coursera_4000_LDA_results.csv')
    
    
    ##### Word2Vec Data Loading
    w2v = gensim.models.Word2Vec.load('w2v_200_dimensions.word2vec')

    with open('w2v_course_with_top_words.pickle', 'rb') as f:
        w2v_course_with_top_words = pickle.load(f)

    return lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words


def lda_w2v_recommender(input_string, lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words):

    #### LDA recommendation
    input_lda_assignment = topic_assignmenter(input_string, lda, mycorpus_dict)

    print 'Working hard to compare your description with destination DNAs...'
    similarity = []
    try:
        LDA_results.drop('cos_similarity', inplace=True, axis=1)
    except:
        pass
    for i in range(LDA_results.shape[0]):
        similarity.append(cos_similarity(np.array(input_lda_assignment),LDA_results.iloc[i,1:lda.num_topics+1]))
    LDA_results['cos_similarity'] = similarity
    LDA_results.sort('cos_similarity', ascending=False, inplace=True)

    lda_recom = []
    for i in range(10):
        lda_recom.append(LDA_results.iloc[i].course_name)


    ##### Word2Vec Section

    ##### Word2Vec Recommendation
    vocab = w2v.vocab.keys()

    w2v_course_list = w2v_course_with_top_words.keys()

    des_1 = [x for x in paragraph_preprocessing(input_string) if x in vocab]
    course_scorer = {}
    for course in w2v_course_list:
        des_2 = [x for x in w2v_course_with_top_words[course] if x in vocab]
        score = w2v.n_similarity(des_1, des_2)
        if isinstance(score, float):
            course_scorer[score] = course
        else:
            course_scorer[0] = course

    sorted_helper = sorted(course_scorer.keys(), reverse=True)
    w2v_recom = []

    for i in range(10):
        w2v_recom.append(course_scorer[sorted_helper[i]])

    return lda_recom, w2v_recom