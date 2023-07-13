import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import requests
from transformers import BertTokenizer, BertModel
import webbrowser
from flask import Flask, render_template, request,url_for, redirect
import sys
import PyPDF2 as pdf
from pdfminer.high_level import extract_text
import csv
import os
import json
import shutil
shutil.rmtree('./uploads')
os.makedirs('./uploads')
from werkzeug.utils import secure_filename

app = Flask(__name__)
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
model = BertModel.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
initialized=False

np_patients = np.loadtxt("MOCK_DATA.csv", delimiter=",", dtype=str)
clinical_trials = pd.read_csv("clinical_trials.csv")
association = pd.read_csv("trial_patient_assoc.csv")
headers=np_patients[0,:]


def init_clinical_trials():
  global clinical_trials
  i=0
  for _, ct in clinical_trials.iterrows():
    ct_id=ct[0]
    patient_index = association.loc[clinical_trials['id'] == ct_id].patient_index[i]
    i+=1
    patient=np_patients[int(patient_index)]
    vector=vectorize(patient)
    clinical_trials.loc[clinical_trials['id'] == ct_id, 'avg_vector'] = str(vector.tolist()).replace(',',' ')

def initial_vector_digest():
  global clinical_trials
  global association
  global initialized
  if not initialized:
    init_clinical_trials()
    vectors=np.empty((np_patients.shape[0], 256))
    vectors[0]=np.zeros(shape=(1,256))
    similarities =pd.DataFrame(columns=['patient_id','clinical_trial_id','similarity'])
    for row in tqdm(np_patients[1:,:]):
      local_sims = get_sims(row)
      similarities = pd.concat(objs=[similarities,local_sims])
      select_trial(row,local_sims['clinical_trial_id'].iloc[0])
    initialized=True
    return similarities
  else:
    return False

def vectorize(row):
  pairs = np.column_stack((headers,row))
  ts = np.array2string(pairs)
  encoded_input = tokenizer(ts, return_tensors='pt')
  output = model(**encoded_input)
  return output.pooler_output.detach().numpy()[0] # Return feature vector
def vectorize_str(txt):
  encoded_input = tokenizer(txt, return_tensors='pt')
  output = model(**encoded_input)
  return output.pooler_output.detach().numpy()[0] # Return feature vector

def get_sims(row,vector_func=vectorize):
  vector = vector_func(row)
  sims=pd.DataFrame(columns=['patient_id','clinical_trial_id','similarity'])
  for _, ct in clinical_trials.iterrows():
    ct_id=ct[0]
    avg_vector_str=ct[5].replace('[','').replace(']','')
    avg_vector=np.fromstring(avg_vector_str,dtype=float, sep=' ')
    num_vectors=ct[6]
    min_age=ct[3]
    max_age=ct[4]
   # if int(row[2])>=int(min_age) and int(row[2])<=int(max_age):
    cos_sim = np.dot(avg_vector, vector)/(np.linalg.norm(avg_vector)*np.linalg.norm(vector))
    temp = pd.DataFrame([[row[0],ct_id, float(cos_sim)]],
                  columns=['patient_id','clinical_trial_id', 'similarity'])
    sims = pd.concat(objs=[sims,temp])

    sims.sort_values(by=['similarity'], ascending=False,inplace=True)
  return sims

def select_trial(row, trial_id):
  global clinical_trials
  global association
  trial=clinical_trials.loc[clinical_trials['id']==trial_id]
  rowid = trial.avg_vector.keys()[0]
  old_avg_vector=np.fromstring(str(trial.avg_vector[rowid]).replace('[','').replace(']',''),dtype=float, sep=' ')
  old_avg_vector*=int(trial.num_patients.iloc[0])
  patient_vector = vectorize(row)
  new_sum_vector = np.add(old_avg_vector,patient_vector)
  new_avg_vector=new_sum_vector/(int(trial.num_patients.iloc[0])+1)
  clinical_trials.loc[clinical_trials['id'] == trial_id, 'avg_vector'] =  str(new_avg_vector.tolist()).replace(',',' ')
  clinical_trials.loc[clinical_trials['id'] == trial_id, 'num_patients'] = trial.num_patients+1

  assoc_row = pd.DataFrame([[trial_id, row[0]]],
                   columns=['clinical_trial','patient_index'])
  association =  pd.concat(objs=[association, assoc_row])
  return association  
      

def makeKeywordDictionary(text):
  oneWordList = ['Date','Weight:','Height:','Gender','Age','Type','Smoker','Consumption','Pressure','Rate','Cholesterol']
  oneWordDict = {'Date': ['birth_date'], 'Weight:': ['weight'],'Height:': ['height'],'Gender':['gender'],'Age':['age'],
                'Type': ['blood_type'],'Smoker':['smoker'],'Consumption':['alcohol_consumption'],'Pressure':['blood_pressure'],
                'Rate': ['heart_rate'],'Cholesterol':['cholesterol_level']}
  delimiters = ['headaches):','allergies:','illness):','regularly:']
  delimitersInfo = {'headaches):':['medical_history','Create'],'allergies:':['allergies','List'],
                  'regularly:':['medication','Create'],'illness):':['medical_history','Medical'],
                  'History':['family_history','Neurological'],'Conditions':['neurological_conditions','Create']}

  infoDict = {}
  splitText = text.split()
  count = len(splitText)
  for i in range(count):
    if splitText[i] in oneWordList:
      key = oneWordDict[splitText[i]][0]
      value = splitText[i+1]
      if key not in infoDict:
         infoDict[key] = []
      infoDict[key].append(value)
    elif splitText[i] == 'Name':
      keyOne = 'first_name'
      keyTwo ='last_name'
      valOne = splitText[i+1]
      valTwo = splitText[i+2]
      if keyOne not in infoDict:
        infoDict[keyOne] = []
      infoDict[keyOne].append(valOne)
      if keyTwo not in infoDict:
        infoDict[keyTwo] = []
      infoDict[keyTwo].append(valTwo)
    elif splitText[i] in delimiters:
      key = delimitersInfo[splitText[i]][0]
      stopPoint = delimitersInfo[splitText[i]][1]
      i = i+1
      while splitText[i] != stopPoint:
        val = splitText[i]
        if key not in infoDict:
          infoDict[key] = []
        infoDict[key].append(val)
        i = i+1
  return infoDict


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower()=='pdf'

if len(sys.argv)>1 and sys.argv[1]=='-w':
  similarities = initial_vector_digest()
  assoc = select_trial(np_patients[17],'J2A-MC-GZGS')

  similarities.to_csv("./similarities.csv",index=False)
  assoc.to_csv("./trial_patient_assoc.csv",index=False)
  clinical_trials.to_csv("./clinical_trials.csv",index=False)


@app.route('/')
def upload_page():
   return render_template('upload.html')

@app.route('/results', methods = ['POST'])
def upload_file():
    f = request.files.get('pdf')
    if f and allowed_file(f.filename):
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
      text = extract_text(f'./uploads/{f.filename}')
      try:
        infoDict = makeKeywordDictionary(text)
        sims=get_sims(json.dumps(infoDict),vectorize_str)
      except:
        sims=get_sims(text,vectorize_str)
      
      vector = str(vectorize_str(json.dumps(infoDict)).tolist()).replace(',',' ')
      df = clinical_trials.merge(sims, left_on=['id'], right_on=['clinical_trial_id'])[['clinical_trial_id','name','desc','min_age','max_age','num_patients','similarity']]
      df.sort_values(by=['similarity'], ascending=False,inplace=True)
      table_html = df.to_html(index=False,table_id="table", justify="center")
      # construct the complete HTML with jQuery Data tables
      # You can disable paging or enable y scrolling on lines 20 and 21 respectively
      # return the html
    return render_template('index.html',tables=table_html,vector=vector)

@app.route('/select/<trial_id>', methods = ['POST'])
def select_trial(trial_id):
  global clinical_trials
  global association
  patient_vector=np.fromstring(str(request.form.get('vector')).replace('[','').replace(']',''),dtype=float, sep=' ')
  trial=clinical_trials.loc[clinical_trials['id']==trial_id]
  rowid = trial.avg_vector.keys()[0]
  old_avg_vector=np.fromstring(str(trial.avg_vector[rowid]).replace('[','').replace(']',''),dtype=float, sep=' ')
  old_avg_vector*=int(trial.num_patients.iloc[0])
  new_sum_vector = np.add(old_avg_vector,patient_vector)
  new_avg_vector=new_sum_vector/(int(trial.num_patients.iloc[0])+1)
  clinical_trials.loc[clinical_trials['id'] == trial_id, 'avg_vector'] =  str(new_avg_vector.tolist()).replace(',',' ')
  clinical_trials.loc[clinical_trials['id'] == trial_id, 'num_patients'] = trial.num_patients+1

  return {'success':True}

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8000, debug=False)





