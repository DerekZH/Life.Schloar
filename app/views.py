from flask import render_template, request, url_for
from app import app
import pymysql as mdb
from a_Model import ModelIt
import pandas as pd
from lda_w2v_recommender.lda_w2v_recommender import lda_w2v_recommender, lda_w2v_loader

#db = mdb.connect(user='root', host='localhost', passwd="chaoshiz", db='world', charset='utf8', port=3307)

lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words, w2v_course_semantic_matrix, w2v_course_semantic_norm, w2v_courses = lda_w2v_loader()

#LDA_results = pd.read_csv('Udemy_Coursera_4000_LDA_results.csv')

@app.route('/')
@app.route('/index')
@app.route('/input')
def cities_input():
    print w2v_courses[0]
    return render_template('input.html', w2v_courses = '*'.join(w2v_courses[:3125]+w2v_courses[3250:]))

#@app.route('/output')
#def cities_output():
#    query_string = request.args.get("ID")
#
#    recommended_courses_lda, recommended_courses_w2v = lda_w2v_recommender(query_string, lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words)
#    
##    LDA_results = pd.read_csv('Udemy_Coursera_4000_LDA_results.csv')
#    courses_lda = []
#    courses_w2v = []
#
#    course_topic_dict = dict(zip(LDA_results.course_name, LDA_results.general_topics))
#    course_provider_dict = dict(zip(LDA_results.course_name, LDA_results.provider))
#
#    for course in recommended_courses_lda:
#        try:
#            courses_lda.append(dict(name=course.encode('utf-8'), general_topics=course_topic_dict[course].encode('utf-8'), provider=course_provider_dict[course].encode('utf-8')))
#        except:
#            continue
#        
#    for course in recommended_courses_w2v:
#        try:
#            courses_w2v.append(dict(name=course.encode('utf-8'), general_topics=course_topic_dict[course].encode('utf-8'), provider=course_provider_dict[course].encode('utf-8')))
#        except:
#            continue
#        
#    return render_template('output.html', courses_lda = courses_lda, courses_w2v = courses_w2v)

@app.route('/output')
def course_output():
    query_string = request.args.get("ID")

    recommended_courses = lda_w2v_recommender(query_string, lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words, w2v_course_semantic_matrix, w2v_course_semantic_norm, w2v_courses)
    
    return render_template('d3.html', recommended_courses = ('*'.join(recommended_courses)).encode('utf-8'), w2v_courses = '*'.join(w2v_courses))
        
