from flask import render_template, request, url_for
from app import app
import pymysql as mdb
import pandas as pd
from lda_w2v_recommender.lda_w2v_recommender import lda_w2v_recommender, lda_w2v_loader

#db = mdb.connect(user='root', host='localhost', passwd="chaoshiz", db='world', charset='utf8', port=3307)

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words, w2v_course_semantic_matrix, w2v_course_semantic_norm, w2v_courses, w2v_course_list_to_match_semantic_matrix, related_topics_dic = lda_w2v_loader()


@app.route('/')
@app.route('/index')
@app.route('/input')
def course_input():
    print len(w2v_courses)
    w2v_courses_for_auto_complete = [course for course in w2v_courses if is_ascii(course)]
    print len(w2v_courses_for_auto_complete)
    return render_template('input.html', w2v_courses = '*'.join(w2v_courses_for_auto_complete))

@app.route('/output')
def course_output():
    query_string = request.args.get("ID")

    try:
        relevance_courses, serendipity_courses = lda_w2v_recommender(query_string, lda, LDA_results, mycorpus_dict, w2v, w2v_course_with_top_words, w2v_course_semantic_matrix, w2v_course_semantic_norm, w2v_courses, w2v_course_list_to_match_semantic_matrix, related_topics_dic, sliding_weight=True)

        recommended_courses = relevance_courses + serendipity_courses

        cols = ['course_name','tsne_1','tsne_2','total_enrollment', 'final_provider', 'final_link', 'final_effort', 'final_price', 'final_img','course_name_for_hover','course_name_for_unhover', 'course_name_with_underscore']
        relevance_courses_for_html = LDA_results.loc[LDA_results.course_name.isin(relevance_courses)][cols].to_dict('records')
        serendipity_courses_for_html = LDA_results.loc[LDA_results.course_name.isin(serendipity_courses)][cols].to_dict('records')

        return render_template('output.html', relevance_courses_for_html = relevance_courses_for_html, serendipity_courses_for_html = serendipity_courses_for_html, recommended_courses = '*'.join(recommended_courses))
    except:
        return render_template('500.html')

@app.route('/world_map')
def course_map_output():
    return render_template('Course_World_Map.html')

@app.route('/slides')
def slides_output():
    return render_template('slides.html')

@app.errorhandler(500)
def page_not_found(e):
    return render_template('500.html'), 500