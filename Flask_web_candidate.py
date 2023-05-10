# install and import required libraries
####################################################################################
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import string
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
wn = nltk.WordNetLemmatizer()
from nltk.corpus import wordnet
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, render_template,jsonify
import os
import pathlib
import matplotlib.pyplot as plt
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
###########################################################################
base_dir = pathlib.Path(__name__).parent.absolute()
#resume_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'client22.csv'))
jobs_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'job_dataset.csv'))
mannual_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'client_data_extraction.csv'))
code_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'Mannual_Extraction.csv'))
##########################################################################
# from candidate_recommendation import *
# from job_recommendation import *
from data_extraction import *
from data_preprocessing import *
from recommendation import *
from chatbot_app import *
from plotly.subplots import make_subplots
user_input_pref =  None
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('home.html')


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    global recommend_job
    global recommend_candidate
    if request.method == 'POST':
        f = request.files['file']
        path_folder = base_dir
        f.save(os.path.join(path_folder,'pdf_file',f.filename))
        path = os.path.join(path_folder,'pdf_file',"*")
        doc_convert_to_pdf(path)
        df = pd.DataFrame(
            columns=['Name', 'Email', 'Mobile', 'Skills', 'Designation', 'Experience_Period', 'education',
                     'countries', 'regions', 'cities', 'Resume_path'])
        def making_df(path_of_list):
            try:
                text = return_text(path_of_list)
                data_in_dic = convert_the_info_dic(text)
                data = extraction_of_Data(path_of_list, data_in_dic, text)
                ########################creating the dataframe###############################
                df.loc[-1] = data
                df.reset_index(drop='index', inplace=True)
            except Exception as e:
                pass
        
        for path_of_list in glob.glob(path):
            if path_of_list.endswith(".pdf"):
                making_df(path_of_list)

            elif path_of_list.endswith(".docx"):
                making_df(path_of_list)
   
        df.to_csv(os.path.join(base_dir,'data_csv','job_recom.csv'), index=False)

        resume_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'job_recom.csv'))
        jobs = givejob(resume_df,jobs_df,5)
        
        path = os.path.join(path_folder,'pdf_file',"*")
        pth = glob.glob(path)
        os.remove(pth[0])

        return render_template('job_recommendation.html',a=jobs)


@app.route('/candidate_recomm/', methods=["GET", "POST"])
def candidate_recomm():
   
    if request.method == "POST":
        # getting input with name = fname in HTML form
        first_name = request.form.get("fname")
        # getting input with name = lname in HTML form
        last_name = request.form.get("lname")
        if first_name and last_name:
            job_data = first_name + last_name
            candidate = cr(resume_df,job_data,3)
            candidate['Resume_path'] = candidate['Resume_path'].apply(lambda x : x.replace('/home/anush/Desktop/flask_name',''))
            return render_template("candidate_download.html", a=zip(candidate['Name'], candidate['Mobile'], candidate['Email'], candidate['Resume_path'] ))
    return render_template("form.html")

# resume parser ==============>

@app.route('/rhome/')
def rhome():
    return render_template('rparser_home.html')


# For analysuis from ground truth     
@app.route('/analysis/')
def analysis():
    num_lst = []
    # count_vector
    def my_count_vector(text):
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text)
        return count_matrix

    # cosine similarity

    def my_textO_similarity(count_matrix):
        x = cosine_similarity(count_matrix)[1][0] * 100
        return x
    for i in ['Name','Mobile','Email','Experience Period','Skills','Education','Location']:
        
        mannual = str(mannual_df[f'{i}']).replace('\[\]','')
        code = str(code_df[f'{i}']).replace('\[\]','')
        text = [mannual,code]

        count_matrix = my_count_vector(text)
        x=my_textO_similarity(count_matrix)
        x = round(x, 2)
        result_name = ('{}'.format(x))
        num_lst.append(result_name)

    n= num_lst
    num_lst = pd.to_numeric(n)
    num_lst
    fig = px.bar(x=['Name','Mobile','Email','Experience Period','Skills','Education','Location'], y=num_lst, color=['Red', 'Blue', 'Green','gray','purple','black','white'],width=700, height=450 )
    #fig= fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False) 
    # Create graphJSON
    fig = fig.update_layout(showlegend=False,title='Show the Data percentage',
      margin=dict(l=100, r=100, t=40, b=20),
        paper_bgcolor="white")

    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen','#00AA00']
    fig2 = go.Figure(data=[go.Pie(labels=['Name','Mobile','Email','Experience Period','Skills','Education','Location'],
                                values=num_lst, pull=[0, 0, 0.2, 0])])
    fig2.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                    marker=dict(colors=colors, line=dict(color='white', width=5)))

    row_matrix=[(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3)]
    labels=['Name','Mobile','Email','Experience Period','Skills','Education','Location']
    values=[87.45,97.00,86.34,65.34,78.34,62,54]
    values = num_lst
    row_matrix=[(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3)]

    fig3 = make_subplots(rows=2, cols=4, specs=[[{'type':'pie'}, {'type':'pie'}, {'type':'pie'}, {'type':'pie'}],[{'type':'pie'}, {'type':'pie'}, {'type':'pie'}, {'type':'pie'}]],subplot_titles=('Name','Email','Mobile','Education','Experience_Period','Skills','Location'),horizontal_spacing=0.2)
    ##########
    for i in range(0,len(labels)):
        fig3.add_trace(go.Pie(labels=[f'{labels[i]}','unmatched'], values=[values[i],100-values[i]], hole=.5), row=row_matrix[i][0], col=row_matrix[i][1])
    ##########
    fig3.update_layout(height=550,showlegend=False,
        margin=dict(l=200, r=200, t=40, b=20),
        font=dict(size=27),
        font_color='#000033',
        title_font_color="black",
        paper_bgcolor="white")
    colors = ['#777FFF', '#CCCCDD']
    fig3.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000022', width=2)))
    # set chart names
    

    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)


    
    # Use render_template to pass graphJSON to html
    return render_template('analysis.html', graphJSON=graphJSON, graphJSON2=graphJSON2,graphJSON3 =graphJSON3)
##############################################################################################################

# Display information
@app.route('/upload_csv')
def upload_csv():
    df = pd.read_csv(os.path.join(base_dir,'data_csv', 'client_data_extraction.csv'))
    Name,Email,Mobile,Skills,Designation,experience_period,education,countries,regions,cities= df['Name'],df['Email'],df['Mobile'],df['Skills'],df['Designation'],df['Experience Period'],df['Education'],df['countries'],df['regions'],df['Location']
    return render_template('index.html',a = zip(Name,Email,Mobile,Skills,Designation,experience_period,education,countries,regions,cities))


@app.route('/download/<path:file_path>', methods=['GET'])
def download(file_path):
    return send_file(file_path, as_attachment=False)
####################################################################################################################################  

#for chatbot uploader
@app.route('/uploader_chatbot', methods=['GET', 'POST'])
def uploader_chatbot():
    global recommend_job
    global recommend_candidate
    if request.method == 'POST':
        file = request.files['file']
        path_folder = base_dir
        file.save(os.path.join(path_folder,'pdf_file',file.filename))
        path = os.path.join(path_folder,'pdf_file',"*")
        doc_convert_to_pdf(path)
        df = pd.DataFrame(
            columns=['Name', 'Email', 'Mobile', 'Skills', 'Designation', 'Experience_Period', 'education',
                    'countries', 'regions', 'cities', 'Resume_path'])
        def making_df(path_of_list):
            try:
                text = return_text(path_of_list)
                data_in_dic = convert_the_info_dic(text)
                data = extraction_of_Data(path_of_list, data_in_dic, text)
                ########################creating the dataframe###############################
                df.loc[-1] = data
                df.reset_index(drop='index', inplace=True)
            except Exception as e:
                pass
        
        for path_of_list in glob.glob(path):
            if path_of_list.endswith(".pdf"):
                making_df(path_of_list)

            elif path_of_list.endswith(".docx"):
                making_df(path_of_list)

        df.to_csv(os.path.join(base_dir,'data_csv','job_recom.csv'), index=False)
        resume_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'job_recom.csv'))
        
        if user_input_pref == None:

            jobs= givejob(df,jobs_df, 1)
            #jobs= givejob(resume_df,jobs_df)
            recommend_job = 0
            path = os.path.join(path_folder,'pdf_file',"*")
            pth = glob.glob(path)
            os.remove(pth[0])
            response= jobs
            print('uploader response', response)
            return jsonify(response)

        else:    
            jobs= givejob(df,jobs_df, user_input_pref)
            recommend_job = 0
            #jobs= givejob(resume_df,jobs_df)
            path = os.path.join(path_folder,'pdf_file',"*")
            pth = glob.glob(path)
            os.remove(pth[0])
            response= jobs
            print('uploader response', response)
            return jsonify(response)
####################################################################################################################
#for chatbot response
@app.route('/get_response', methods=['POST'])
def get_chatbot_response():
    global recommend_job
    global recommend_candidate
    global user_input_pref
    global user_input_pref2
    response = None
    user_input = request.form['user_input'].lower()
    if (("job" in user_input) and ('other' not in user_input)) and ("job" in user_input) and ('more' not in user_input):
        recommend_job = 1
        response= 'Please enter your skills or upload your resume'
        return jsonify(response)

    if recommend_job == 1:
        recommend_job = 0    
        response = givejob2([user_input],jobs_df, 1)
        return jsonify(response) 


    if (("candidate" in user_input) and ('other' not in user_input)) and ("candidate" in user_input) and ('more' not in user_input):
        recommend_candidate = 1
        recommend_job = 0
        response = 'Please enter job requirement and description'
        return jsonify(response)

    if recommend_candidate == 1:
        response = cr(resume_df,user_input,1)
        response['Resume_path'] = response['Resume_path'].apply(lambda x : x.replace('/home/anush/Desktop/chat_bot',''))
        print("Chatbot:", response)
        recommend_candidate = 0
        return jsonify(response.to_json())

    if (('other' in user_input) and ('job' in user_input)) or (('more' in user_input) and ('job' in user_input)) or (('another' in user_input) and ('job' in user_input)):
        recommend_job = 2
        response= 'Please provide number of recommendation you want'
        return jsonify(response)

    if (('other' in user_input) and ('candidates' in user_input)) or (('more' in user_input) and ('candidates' in user_input)) or (('another' in user_input) and ('candidates' in user_input)):
        recommend_candidate=2
        response= 'Please provide number of recommendation you want'
        return jsonify(response)

    if  recommend_job == 2:
        recommend_job = 3
        user_input_pref = request.form['user_input'].lower()
        user_input_pref = int(user_input_pref)
        response = 'Please enter your skills or upload your resume'
        return jsonify(response)

    if recommend_job == 3:
        response = givejob2([user_input],jobs_df, user_input_pref)
        recommend_job = 0
        return jsonify(response) 

    if  recommend_candidate == 2:
        recommend_candidate=3
        user_input_pref2 = request.form['user_input'].lower()
        user_input_pref2 = int(user_input_pref2)
        response = 'Please enter job description'
        return jsonify(response)



    if recommend_candidate == 3:
        response = cr(resume_df,user_input,user_input_pref2)
        response['Resume_path'] = response['Resume_path'].apply(lambda x : x.replace('/home/anush/Desktop/chat_bot',''))
        print("Chatbot:", response)
        recommend_candidate = 0
        return jsonify(response.to_json())


        #return render_template("chatbot.html", response= response,rec = recommend_job, rec2 =4, a=zip(response['Name'], response['Mobile'], response['Email'],response['Resume_path'] ))        
    for intent in intents:
        for pattern in intents[intent]["patterns"]:
            if pattern in user_input:
                #response = intents[intent]["responses"][random.randint(0,len(responses) -1 )]
                response = intents[intent]["responses"][0]
                return jsonify(response)

    if not response:
        response = get_response(user_input)
    return jsonify(response)

# top 10 job recommendation function call in fastapi ========================================
def recommend_jobs(resume_df, jobs_df, n=10):
    """
    Given a resume dataframe and a jobs dataframe, returns the top n recommended jobs
    """
    # code to compute job recommendations
    recommended_jobs = []
    for i in range(n):
        recommended_jobs.append(f"Job {i+1}")
    return recommended_jobs

##########################################################################################################################

@app.route('/chatbot',methods = ['GET', 'POST'])
def chatbot():
    rec = 1
    return render_template("chatbot2.html", rec= rec)
######################################################################################################################################
     
if __name__ == '__main__':
    app.run(debug = True)
