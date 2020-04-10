from flask import Flask, render_template, request, url_for


import io
import os
import glob

import sys
sys.path.append('../')

import poc
from poc import *


app = Flask(__name__)


contracts = []

@app.route('/', methods=['GET', 'POST'])
def index():
  return render_template('index.html')


@app.route('/test_contracts/', methods=['GET', 'POST'])
def test_contracts():
  print('GETTING CONTRACTS...')
  
  # create contract files
  filenames  = []
  parent_dir = os.getcwd()+'/data'
  
  for pdf_file in glob.glob(os.path.join(parent_dir, '*.pdf')):
    filenames.append(pdf_file)


  for fn in filenames:
    print(fn)
    c=test(pdf=fn, savejson=True, savetxt=False)
    contracts.append(c)
  
  print('DONE GETTING CONTRACTS...')
  return 'Contract Click.'


@app.route('/test_search/', methods=["GET", "POST"])
def test_search():
  print ('START SEARCH...')

  searchStr=''
  if request.method == "POST":
     searchStr= request.form["searchstr"]
  elif request.method=='GET':
       searchStr=request.args.get('searchstr', '')
  else:
    print('unhandled request: ', request.method)
    return 'Search Click.'

  print('FORM TEXT:', searchStr)
    
  if len(contracts) == 0:
      test_contracts()

  s = "{'legal': 'anti-bribery', 'financial': 'escrow','financial': 'perpetual','SLA': 'breach','SLA': 'uptime'}"
  df = doSearch(contracts, s)
  print (df)
    
  print ('DONE START SEARCH...')
  return 'Search Click.'


if __name__ == '__main__':
  app.run(debug=True)