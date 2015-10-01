import pandas as pd
import pickle
import gensim
import sys
import numpy as np
import nltk
from helpers.helpers import *

def get_cos_similariy(course_name):
    try:
        return semantic_score_rev_dic[course_name]
    except:
        return 0

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

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

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

#    with open('Combined_Udemy_Coursera_Reviews.pickle', 'rb') as f:
#        Udemy_Coursera_combined_train = pickle.load(f)

    with open('Combined_Udemy_Coursera_Edx_Reviews.pickle', 'rb') as f:
        Udemy_Coursera_combined_train = pickle.load(f)

    mycorpus = MyCorpus(Udemy_Coursera_combined_train)
    mycorpus.dictionary.filter_extremes()
    mycorpus.dictionary.items()
    mycorpus_dict = gensim.corpora.dictionary.Dictionary()
#    mycorpus_dict= mycorpus_dict.load('LDA_Udemy_Coursera_20_topics_gensim_dict.dict')
    mycorpus_dict= mycorpus_dict.load('LDA_Udemy_Coursera_Edx_26_topics_gensim_dict.dict')
    
    lda = gensim.models.ldamulticore.LdaMulticore(id2word=mycorpus.dictionary.id2token)
#    lda = lda.load('lda_full_20_topics.lda')
    lda = lda.load('LDA_Udemy_Coursera_Edx_26_topics.lda')
#    LDA_results = pd.read_csv('Udemy_Coursera_4000_LDA_results.csv')
    LDA_results = pd.read_csv('LDA_results_CEU_26_topics.csv')
    
    
    ##### Word2Vec Data Loading
    w2v = gensim.models.Word2Vec.load('w2v_300_dimensions_CEU_with_wiki_data_added.word2vec')

    with open('w2v_course_with_top_words_CEU_wiki_data_added.pickle', 'rb') as f:
        w2v_course_with_top_words = pickle.load(f)
    
    with open('w2v_course_semantic_matrix_CEU.pickle', 'rb') as f:
        w2v_course_semantic_matrix = pickle.load(f)
    
    with open('w2v_course_semantic_norm_CEU.pickle', 'rb') as f:
        w2v_course_semantic_norm = pickle.load(f)

    with open('w2v_courses_CEU.pickle', 'rb') as f:
        w2v_courses = pickle.load(f)
        
    return lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words, w2v_course_semantic_matrix, w2v_course_semantic_norm, w2v_courses


def lda_w2v_recommender(input_string, lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words, w2v_course_semantic_matrix, w2v_course_semantic_norm, w2v_courses, num_to_recommend = 5):

    couse_index_padding = 0
    if input_string in w2v_course_with_top_words.keys():
        input_lda_assignment = LDA_results.loc[LDA_results.course_name == input_string].iloc[:,1:lda.num_topics+1].values.T
        couse_index_padding = 1
        print 'dada'
    else:
        input_lda_assignment = topic_assignmenter(input_string, lda, mycorpus_dict)
        input_lda_assignment = np.array(input_lda_assignment)
    print 'Working hard to compare your description with course DNAs...'
    print input_lda_assignment.shape

    try:
        LDA_results.drop('cos_similarity', inplace=True, axis=1)
    except:
        pass

    topic_matrix = LDA_results.iloc[:,1:lda.num_topics+1].values

    topic_matrix_norm = []
    for i in range(topic_matrix.shape[0]):
        topic_matrix_norm.append(np.linalg.norm(topic_matrix[i,:]))

    topic_matrix_norm = np.array(topic_matrix_norm)

    similarity = topic_matrix.dot(input_lda_assignment).reshape((topic_matrix.shape[0],1)) / topic_matrix_norm.reshape((topic_matrix_norm.shape[0],1)) / np.linalg.norm(input_lda_assignment)
    print topic_matrix.dot(input_lda_assignment).shape
    print topic_matrix_norm.reshape((topic_matrix_norm.shape[0],1)).shape
    print np.linalg.norm(input_lda_assignment)
    print similarity.shape

    LDA_results['cos_similarity'] = similarity
    LDA_results.sort('cos_similarity', ascending=False, inplace=True)

    lda_recom = []

    off_set = 0
    ######## code below is not a complete fix, need to update
    for i in range(num_to_recommend):
        if is_ascii(LDA_results.iloc[i+off_set].course_name):
            lda_recom.append(LDA_results.iloc[i+off_set+couse_index_padding].course_name)
        else:
            off_set += 1
            lda_recom.append(LDA_results.iloc[i+off_set+couse_index_padding].course_name)

    #    print '\n\nLDA results'        
    #    for i in range(100):
    #        print LDA_results.iloc[i].course_name, LDA_results.iloc[i].cos_similarity
    #    print '\n\n'

    ##### Word2Vec Section

    ##### Word2Vec Recommendation
    vocab = w2v.vocab.keys()

    couse_index_padding = 0
    if input_string in w2v_course_with_top_words.keys():
        des_1 = [x for x in w2v_course_with_top_words[input_string] if x in vocab]
        couse_index_padding = 1
    else:
        des_1 = [x for x in paragraph_preprocessing(input_string) if x in vocab]


    des_1_norm = np.linalg.norm(w2v_get_semantics_for_list_of_words(w2v, des_1))
    similarity_results = w2v_course_semantic_matrix.dot(w2v_get_semantics_for_list_of_words(w2v, des_1))/w2v_course_semantic_norm/des_1_norm
    semantic_score_dic = dict(zip(similarity_results, w2v_courses))

    w2v_recom = []
    off_set = 0  # off_set is used to exclude unicode strings
    for i in range(couse_index_padding, couse_index_padding+num_to_recommend):
        if is_ascii(semantic_score_dic[sorted(semantic_score_dic.keys(), reverse=True)[i+off_set]]):
            w2v_recom.append(semantic_score_dic[sorted(semantic_score_dic.keys(), reverse=True)[i+off_set]])
        else:
            off_set += 1
            w2v_recom.append(semantic_score_dic[sorted(semantic_score_dic.keys(), reverse=True)[i+off_set]])

    semantic_score_rev_dic = {}
    for key, value in semantic_score_dic.items():
        semantic_score_rev_dic[value] = key

    LDA_results['w2v_cos_similarity'] = LDA_results.course_name.apply(get_cos_similariy)
    LDA_results['weighted_cs'] = (LDA_results.cos_similarity + LDA_results.w2v_cos_similarity)/2
    
    recom_results = LDA_results.sort('weighted_cs', ascending=False).course_name.iloc[:num_to_recommend+10].values
    recom_results = [x for x in recom_results if x != input_string]
    recom_results = [x for x in recom_results if is_ascii(x)]
    return recom_results[:num_to_recommend]