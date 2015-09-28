import pandas as pd
import pickle
import gensim
import sys
import numpy as np
import nltk
from helpers.helpers import *

def w2v_get_semantics_for_list_of_words(model, list_of_words):
    for word in list_of_words:
        try:
            vec = vec + model[word]
        except:
            try:    #deal with words that are excluded from w2v model because of low frequency
                vec = model[word]
            except:
                pass

    vec = vec/float(len(list_of_words))
    
    return vec


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
    w2v = gensim.models.Word2Vec.load('w2v_300_dimensions_wiki_data_added.word2vec')

    with open('w2v_course_with_top_words_wiki_data_added.pickle', 'rb') as f:
        w2v_course_with_top_words = pickle.load(f)
    
    with open('w2v_course_semantic_matrix.pickle', 'rb') as f:
        w2v_course_semantic_matrix = pickle.load(f)
    
    with open('w2v_course_semantic_norm.pickle', 'rb') as f:
        w2v_course_semantic_norm = pickle.load(f)

    with open('w2v_courses.pickle', 'rb') as f:
        w2v_courses = pickle.load(f)
        
    return lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words, w2v_course_semantic_matrix, w2v_course_semantic_norm, w2v_courses


def lda_w2v_recommender(input_string, lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words, w2v_course_semantic_matrix, w2v_course_semantic_norm, w2v_courses, num_to_recommend = 5):


    #### LDA recommendation
    input_lda_assignment = topic_assignmenter(input_string, lda, mycorpus_dict)
    input_lda_assignment = np.array(input_lda_assignment)
    print 'Working hard to compare your description with course DNAs...'

    try:
        LDA_results.drop('cos_similarity', inplace=True, axis=1)
    except:
        pass
    topic_matrix = LDA_results.iloc[:,1:lda.num_topics+1].values
    similarity = topic_matrix.dot(input_lda_assignment)
    
    LDA_results['cos_similarity'] = similarity
    LDA_results.sort('cos_similarity', ascending=False, inplace=True)

    lda_recom = []
    for i in range(num_to_recommend):
        lda_recom.append(LDA_results.iloc[i].course_name)


    ##### Word2Vec Section

    ##### Word2Vec Recommendation
    vocab = w2v.vocab.keys()

    couse_index_padding = 0
    if input_string in w2v_course_with_top_words.keys():
        des_1 = [x for x in w2v_course_with_top_words[input_string] if x in vocab]
        couse_index_padding = 1
    else:
        des_1 = [x for x in paragraph_preprocessing(input_string) if x in vocab]

    print des_1
    
    des_1_norm = np.linalg.norm(w2v_get_semantics_for_list_of_words(w2v, des_1))
    similarity_results = w2v_course_semantic_matrix.dot(w2v_get_semantics_for_list_of_words(w2v, des_1))/w2v_course_semantic_norm/des_1_norm
    semantic_score_dic = dict(zip(similarity_results, w2v_courses))

    w2v_recom = []
    for i in range(couse_index_padding, couse_index_padding+num_to_recommend):
        w2v_recom.append(semantic_score_dic[sorted(semantic_score_dic.keys(), reverse=True)[i]])
        print sorted(semantic_score_dic.keys(), reverse=True)[i]

    return lda_recom, w2v_recom