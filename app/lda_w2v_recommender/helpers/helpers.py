import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.collections import PatchCollection
import gensim
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
import nltk
import enchant

def cos_similarity(a, b):
    return np.dot(a,b) / float(np.linalg.norm(a) * np.linalg.norm(b))
    
def paragraph_stemmer(string): #and non english word remover
    d = enchant.Dict("en_US")
    porter = nltk.stem.porter.PorterStemmer()
    string = string.replace('\n',' ').replace(',',' ').replace('.',' ').replace('!',' ').replace('?',' ').replace('\r',' ').split(' ')
    new_string = []
    removed_foreign_words = []
    for item in string:
        try:
            if d.check(item):
                new_string.append(porter.stem(item.encode('utf-8').lower()))
            else:
                removed_foreign_words.append(item)
        except:
            pass
    return ' '.join(new_string), removed_foreign_words

    
def paragraph_preprocessing(paragraph):
    paragraph, removed_words = paragraph_stemmer(paragraph)
    paragraph = tokenize(paragraph)
    return paragraph

def topic_assignmenter(paragraph, lda, mycorpus_dict):
    paragraph = paragraph_preprocessing(paragraph)
    result = lda[mycorpus_dict.doc2bow(paragraph)]
    num_topics = lda.num_topics
    test_topic_assignment = [0 for x in range(num_topics)]
    for item in result:
        test_topic_assignment[item[0]] = item[1]
    return test_topic_assignment

def tokenize(text):
    my_stop_words = ['place', 'great', 'nice', 'good', 'time', 'visit', 'day', 'best', 'tour', 'lovely', 'excellent', 'worth', 'hi','learn','inform','work','new']
    my_stop_words_2 = ['em','el', 'en', 'tem', 'por', 'hors', 'il', 'et', 'es', 'passeio', 'le', 'che', 'la', 'lo', 'muy', 'per', 'muito', 'uma', 'ba', 'para', 'de', 'da', 'bu', 'hi', 'que', 'bem', 'amalfi', 'st', 'um', 'una', 'un', 'bom', 'na', 'com', 'se', 'con']
    porter = nltk.stem.porter.PorterStemmer()

    stop_words = nltk.corpus.stopwords.words('english') + my_stop_words + my_stop_words_2
    return [token for token in gensim.utils.simple_preprocess(text) if token not in stop_words]


def matching_scorer(dd_result, des_topic_fingerprint):
    des_topic_fingerprint = des_topic_fingerprint.values.tolist()
    matching_score = 0
    for i in range(len(dd_result)):
        matching_score += dd_result[i] * des_topic_fingerprint[i]
    return matching_score





def vector_rotator(x, theta):
    from math import pi, cos, sin
    theta_arc = 2.*pi*theta / 360
    rotation_matrix = np.array([[cos(theta_arc), -sin(theta_arc)],[sin(theta_arc), cos(theta_arc)]])
    return np.dot(rotation_matrix, x)
    
    
def spider_chart(ax, topic_contributions = [[0.4,0.4,0.9,0,0,0.5,0,0]], topics = ['data' for i in range(8)], add_axis=False, annotate_topics=False):
    n_topics = len(topics)
    
    x = np.array([[1,0]]).T
    ys = []    # grid vectors to plot
    z = np.array([1,0]) * np.linalg.norm(x)     # adding a horizontal line to denote axis

    # plotting canvas background    
    # rotate topics to its direction
    for i in range(n_topics):
        ys.append(vector_rotator(x, 360/float(n_topics)*i))
    ys.append(ys[0])

    b = np.array(ys).reshape(n_topics+1,2)
    percent_list = [0.25*pc for pc in range(5)]

    ax.plot(b[:,0], b[:,1],'-', color='#399E5A', linewidth=1)
    for percent in percent_list:
        ax.fill(percent*b[:,0], percent*b[:,1],'-', color='#399E5A', alpha=0.3*(1-percent)) 
        
        if add_axis:
            ax.plot(percent*z[0], percent*z[1],'|k')  # here z is the vertical axis tick
            if percent != 1:
                ax.annotate(percent,(percent*z[0]-0.04, percent*z[1]-0.1), alpha=0.4)
 
    for i in range(b.shape[0]):   # adding diagonal line for each topic direction
        ax.plot([0,b[i,0]], [0,b[i,1]], '-', linewidth=0.5,color='#399E5A', alpha=0.3)


    if annotate_topics:
        # topic text annotation
        hfont = {'fontname':'Arial Rounded Bold','fontsize':14,'color':'#121212'} 
        for i in range(b.shape[0]-1): 
            tiny_tweak = 0.022
            text_rotation = 360/float(n_topics)*i-90
            if b[i,:][1]<0:
                text_rotation += 180

            additional_offset = np.array([0,0])*tiny_tweak * 1
            
            if b[i,:][1]<0:
                additional_offset = additional_offset + np.array([0, -2*tiny_tweak])
            if b[i,:][0]<0:
                additional_offset = additional_offset + np.array([-1.6*tiny_tweak*len(topics[i]),0])
            if b[i,:][0]==0:
                additional_offset = additional_offset + np.array([-0.8*tiny_tweak*len(topics[i]),0])

            ax.annotate(topics[i],tuple(b[i,:] + additional_offset), fontsize=14)#, rotation = text_rotation, )
            
            
    
    # plotting actual data
    
    colors = ['#FB5B1C','#9F2042','#F62DAE','#26532B','#5ABCB9','#50808E','#A33B20','#462D42',]
    for ind, topic_contribution in enumerate(topic_contributions):
        #topic_contribution = [max(tc,0.0002) for tc in topic_contribution]
        topic_contribution = [float(tc)/sum(topic_contribution) for tc in topic_contribution]
        topic_contribution = [tc if tc>0.03 else 0 for tc in topic_contribution]        
        if max(topic_contribution) < 0.6:
            topic_contribution = [float(tc)/max(topic_contribution)*0.6 for tc in topic_contribution]
        nonzero_index = np.where(np.array(topic_contribution)!=0)[0]

        y2s = []   # topic contribution vectors to plot

        # rotate topics to its direction
        for i in range(n_topics):
            y2s.append(vector_rotator(topic_contribution[i]*x, 360/float(n_topics)*i))

        _y2s = [y for y in y2s if y[0]**2+y[1]**2 != 0]  #removing zero values from topic
        
        y2s = [_y2s[0]]
        for i in range(1,len(_y2s)):
            if nonzero_index[i]-nonzero_index[i-1] > len(topic_contribution)/2.:
                y2s.append(np.array([[0,0]]).T)
            y2s.append(_y2s[i])

        if nonzero_index[0]+len(topic_contribution)-nonzero_index[-1] > len(topic_contribution)/2.:
            y2s.append(np.array([[0,0]]).T)

        y2s.append(y2s[0])
        c = np.array(y2s).reshape(len(y2s),2)

        ax.plot(c[:,0], c[:,1],'-', color=colors[ind+1], linewidth=2, alpha=0.7)    
        ax.fill(c[:,0], c[:,1],'-', color=colors[ind+1], alpha=0.3)
#        box = mpatch.FancyBboxPatch((0,0),0.05,0.1,boxstyle=mpatch.BoxStyle("Round", pad=0.02)) 
#        ax.add_patch(box)


    ax.axis('equal')
    ax.axis([-1.5,1.5,-1.5,1.5])
    ax.axis('off')
    
    

    
