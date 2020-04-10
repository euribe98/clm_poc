#!/usr/bin/env python
# coding: utf-8


class Party():
    
    address = ''
    _type = ''
    
    def __init__(self, sp):
        self._span = sp
        self.name =  sp.text
        self.spsent = sp.sent[sp.sent.start : sp.sent.end]
     
    
    def findParty(parties, name):
        #_parties = [x.name.lower() for x in parties]
        
        for p in parties:
            if p.name.lower() == name.lower():
                return p
        return None
    

    
    def printParty(parties):
        print([(p.name) for p in parties])
        
         


# In[2]:


# contract class using spacy


import pandas as pd
import io
import os
import glob

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import codecs
import json
import re


from pprint import pprint
from collections import Counter

import spacy
from spacy import displacy
import en_core_web_sm
from spacy import displacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler 

import usaddress
import nltk



class Contract():
    """ Class representation of a contract """
    
    SMALL_MOD = 'en_core_web_sm'
    LARGE_MOD = 'en_core_web_lg'
    DISPLAY_OPTIONS = {"compact": True, "color": "blue", 'distance':140}
    
    
    # contract attributes  
    orgtext = ''    # raw text
    text = ''       # cleaned text
    title =''       # contract title
    sentences = []  # tokenized sentences, punctuation delimited
    wordtokens = [] # tokenized words
    lines = []      # tokenized lines, \n delimited
    parties = []    # Party objects
    sections = []   # sections
        
    ## data model, dictionary for json 
    contract = {} # root
    party = {}
    section = {}
    service = {}

    
    ## construction
    def __init__(self, txtfile='', pdffile=''):
        
        """ Initializes a Contract object 
    
        Parameters:
        txtfile (str): contract text file
        pdffile (str): contract pdf file
        Returns: None
        
        """
        
        self.pdffile = pdffile
          
        # load english model and return lang oject
        #self.nlp = spacy.load(self.SMALL_MOD)  # problems in some environments?
        self.nlp = en_core_web_sm.load()
        
        ### TODO: https://spacy.io/usage/processing-pipelines/
        #merge_ents = self.nlp.create_pipe("merge_entities")
        #self.nlp.add_pipe(merge_ents)
        #merge_nps = nlp.create_pipe("merge_noun_chunks")
        #self.nlp.add_pipe(merge_nps)

        
        # set raw text 
        if len(txtfile) > 0:
            f = open(txtfile, 'r')  # read-from text file (debug only)
            self.orgtext = f.read()  
            
        elif len(pdffile) > 0:
            self.orgtext = self.convert_pdf_to_txt (pdffile) # read pdf and convert to text
            
        self.text = self.orgtext
        
        # preprocessing
        self.preprocess()
        
        # tokenize 
        self.lines = self.text.splitlines()             # lines
        self.sentences = self.sent_segment(self.text)   # sentences
        self.wordtokens = self.get_words(self.text)     # word tokens
        
        # add address entity from preample
        txt= self.sentences[0].replace('\n', ' ')  # line feeds causes problems: TODO:
        self.addAddressEntity(txt) 
        
        # set up parties
        self.parties = self.getParties (txt)
        
        # set up sections
        self.sections = self.getSections(self.orgtext)
    

        
    # this logic is not ideal, assumes positional  
    def updateAddress(self, doc):
        #print('\n', [(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents])

        orgs = [ent for ent in doc.ents if ent.label_ == 'ORG']
        #print('\norgs:', [(ent.text, ent.label_, ent.ent_id_) for ent in orgs]) 

        addr = [ent for ent in doc.ents if ent.label_ == 'ADDRESS']
        #print('\naddr:', [(ent.text, ent.label_, ent.ent_id_) for ent in addr]) 

        #print('\nparties:', [(p.name) for p in c.parties]) # ok

        i = 0;
        for o in orgs:
            #print('\nprocessing: ', o.text)
            p = Party.findParty(self.parties, o.text)

            if p != None:
              p.address = addr[i].text
              if 'metricstream' in o.text.lower():
                 p._type = 'promisee'
              else:
                  p._type = 'customer' 
            i=i+1
            
    def getParties(self, text):
        self.parties = []
        
        doc = self.get_doc(text)
        orgs= [ent for ent in doc.ents if ent.label_ == 'ORG']

        for sp in orgs:
            p = Party(sp)
            if "United\nStates" not in p.name: ## TODO: special case
               self.parties.append(p)
        
       
        # update party with address
        self.updateAddress(doc)
        ''''
        # this needs to be explored further, but it does not generalize across contracts
        orgs= [ent for ent in doc.ents if ent.label_ == 'ADDRESS']
       
        for sp in orgs:
            #print ('\n', sp.text, '-> ', sp.start, sp.end)
            spsent = sp.sent[sp.sent.start : sp.sent.end] # will be 0 if it's at the end of the sentence

            newsent = str(spsent).replace('\n', ' ').replace('“', ' ').replace('”', ' ')
            d = self.get_doc(newsent)
            ents = [(ent.text, ent.label_) for ent in d.ents]

            if len(newsent) == 0:
                p = Party.findParty(self.parties, 'METRICSTREAM') ## TODO: special case
                if p != None:
                    p.address = sp.text
                    p._type = 'promisee'
            else:
                for e in ents:
                    #print(e[0], e[1])
                    p = Party.findParty(self.parties, e[0])
                    if p != None:
                        p.address = sp.text
                        p._type = 'customer' ## TODO: figure this out
        '''

        return self.parties


    # Tokenize a document into topical sections using the TextTiling algorithm. 
    # This algorithm detects subtopic shifts based on the analysis of lexical co-occurrence patterns.
    def getSections(self, text, writetofile=False):

        def cleanNumbers(lines):
            regx = 'Page\s+[0-9]+\s+of\s+[0-9]+'
            new = ''
            for s in lines:
                s=s.strip()

                if len(s)==0 or s.isnumeric() or re.match(regx, s):  
                    continue

                new+=s+'\n'
            return new


        tt = nltk.tokenize.TextTilingTokenizer()
        sections = tt.tokenize(text)


        para = []
        for sec in sections:
            newsec = self.clean_empty_lines(sec)
            lines = newsec.split("\n")
            newsec = cleanNumbers(lines)  # clean
            para.append(newsec)

        if writetofile:
            with open('./sections.txt', 'w') as f3:
              for s in para:
                f3.write(s + '<EOL>\n\n')
        return para


    
    def addAddressEntity(self, text):
        addrlist = self.getAddressList(text)

        cnt=0
        for a in addrlist:
            cnt=cnt+1
            #print(a)
            ruler = EntityRuler(self.nlp, overwrite_ents=True)
            ruler.name = "addr_"+str(cnt)
            pattern = [{"label": "ADDRESS", "pattern": a}]
            
            ruler.add_patterns(pattern)
            
            try:
                self.nlp.add_pipe(ruler)
            except ValueError as ve:
               self.nlp.remove_pipe(ruler.name)
               self.nlp.add_pipe(ruler)
           
            
        
    def preprocess(self):
       
        # clean text
        self.text = self.clean_abbrev(self.text)       # remove dots in abbreviations
        self.text = self.clean_text(self.text)         # remove special characters
        self.text = self.clean_empty_lines(self.text)  # remove empty lines
    
        # add Definition Entity Recognizer
        ruler = EntityRuler(self.nlp, overwrite_ents=True)
        ruler.name = 'definition'
        patterns = [ {"label": "DEFINITION", "pattern": [{"ORTH": "("}, {'IS_ALPHA': True}, {"ORTH": ")"}]},
                     #{"label": "DEFINITION", "pattern": [{"LOWER": "effective"}, {"LOWER": "date"}]},
                     #{"label": "GPE", "pattern": [{"LOWER": "united\n"}, {"LOWER": "states"}]}
                   ]
        
        ruler.add_patterns(patterns)
        try:
            self.nlp.add_pipe(ruler)
        except ValueError as ve:
            self.nlp.remove_pipe(ruler.name)
            self.nlp.add_pipe(ruler)
          
           
        
            
    def get_words(self, txt):
        """ Creates word tokens from text

        Parameters:
        txt (str): text to tokenize

        Returns:
        list: list of tokenized words

        """
        doc = self.get_doc(txt)
        tokens = []
        for sent in doc.sents:
            tokens.append([token.text for token in sent]) # for each word in sentence
        return tokens
    
    
    def getAddressList(self, text):
        subs=['AddressNumber', 'StreetNamePreDirectional', 'StreetName',
              'StreetNamePostType','PlaceName','StateName','ZipCode']

        parsed = usaddress.parse(text)
        parsed = [x for x in parsed if x[1] in subs ]

        addrlist = []
        addr=''
        for x in parsed:
            val = x[0]
            typ =  x[1]
            
            if typ != 'ZipCode':
                addr+=val+' '
                #print(val,'-> ', typ)
            else:
                addr+=val+' '
                addrlist.append(addr.strip())  
                addr=''

        return addrlist
           

    def getAddress(self, text):
        
        """ parse text for address 
    
        Parameters:
        text (str): text to parse
        Returns: address string
        
        """
        
        doc = self.nlp(text)

        # for some reason zipcode 60601 is not an entity, so we can't filter by entity type
        #print([(ent.text, ent.label_) for ent in doc.ents])
        #str = ['CARDINAL', 'LOC', 'GPE', 'FAC']
        #addrlist = [ent.text for ent in doc.ents if ent.label_ in str]
        #addr = " ".join(addrlist)
        #parsed = usaddress.parse(addr)

        parsed = usaddress.parse(text)
        parsed = [x for x in parsed if x[1] != 'Recipient'] # filter out recipient

        addr=''
        for x in parsed:
            val = x[0]
            typ =  x[1]
            addr+=val+' '
            #print(val,'-> ', typ)

        if len(addr) > 0:
            addr = addr.replace('.', '').strip()
        return addr
    
    
    def addParties (self):
        plist = []

        for p in self.parties:
            #print (p.name, '=> ', p.address)
            e = ['name,'+ p.name, 'type,'+p._type, 'address,' + p.address]
            p = self.add_entity(e)
            plist.append(p)

        # update model
        self.updateModelList('party', plist)
    

    def addSections(self):
        slist = []
        
        for sec in self.sections:
            txt = sec.split('\n')
            typ = txt[0]
            text = '\n'.join(txt[1:])

            s = self.add_entity(['type,'+ typ, 'text,'+text]) 
            slist.append(s)
            
        self.updateModelList('section', slist)

    

    
    def add_entity(self, keyvalues):
        return dict(item.split(",",1) for item in keyvalues)
    
    
       
    def updateModel(self, key, value):
         """ adds attributes to contract dictionary 
         Parameters:
         key (str): key name 
         value(str): value
         Returns: None
         """
         self.contract[key]  = value
        
            
    def updateModelList(self, key, values):
         self.contract[key]  = values
        
        
    
    def getEffectiveDate(self, doc):
        ### TODO: not perfect
        dateslist = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
        return dateslist[0]
    
        
    def getJson(self, savetofile=False):
        """ Converts contract to json object 
    
        Parameters:
        savetofile (boolean): whether to save to json file (optional)
        Returns: json string
        
        """
    
        ## data model
        self.updateModel('filename', self.pdffile)
        self.updateModel('title', self.lines[0])
        
        effdate = self.getEffectiveDate (self.nlp(self.sentences[0]))
        self.updateModel('effective date', effdate) 
        
        
        # add parties  
        self.updateModel('party', self.party)  
        self.addParties()
        
        # add sections
        self.updateModel('section', self.section) 
        self.addSections()
      
        # service 
        self.updateModel('service', self.service)     
        
        self.updateModel('text', self.orgtext)
   
        # Serializing json  
        json_str = json.dumps(self.contract, indent = 4) 
        
        # Writing to sample.json 
        if savetofile:
            filename =  getTxtFileName(self.pdffile, '.json') 
            self.save_to_file(filename, json_str) 
        return json_str
    
    
    # read json
    def printJson(self, contract, getjson=False):
        """ print json representation of the contract object

        Parameters:
        contract (str): contract dictionary 
        getjson (bool): whether to get the json object

        Returns: None
        
        """
        if getjson:
            self.getJson(False)
            
        for key, value in contract.items():
            print("{k}: {val}".format(k=key, val=value))
    

    def clean_abbrev(self, txt):
        """ remove dots in abbrevations from text
    
        Parameters:
        txt (str): text to evaluation
        Returns: text with cleaned abreviations
        
        """
        # need toget rid of '.' since spacy treat that as sentence delimieter
        abbrevs={'U.S.A':'USA', 'INC.':'INC '}    # space is needed, for spacy named-entity
        for abbrev in abbrevs:
            #clean= txt.replace(abbrev, abbrevs[abbrev])
            clean = re.sub('(?i)'+re.escape(abbrev), lambda m: abbrevs[abbrev], txt) # case insensitive

        return clean
    

    def clean_text(self, txt):
        """ text cleaning
    
        Parameters:
        txt (str) : text to clean 
        Returns: cleaned text
        
        """
       
        # commas (,) cause issues. default ner seems to be using it identify ORGS 
        clean = re.sub(r'[^a-zA-Z0-9.(),“”\s]+', ' ', txt)  # keep periods for sentence tokenizing
        return clean
    

    def clean_empty_lines(self, txt):
        """ remove empty lines from the text
    
        Parameters:
        txt (str) : text to clean
        Returns: cleaned text
        
        """
        lines = txt.split("\n")
        non_empty_lines = [line for line in lines if line.strip() != ""]

        without_empty_lines = ""
        for line in non_empty_lines:
            without_empty_lines += line.strip() + "\n"
            
        return without_empty_lines
    

   
    def convert_pdf_to_txt(self, pdf):
        """ convert pdf to text and return the text
    
        Parameters:
        pdf (str) : contract pdf file
        Returns: contract raw text
        
        """
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        
        laparams = LAParams()
        layout = LAParams(all_texts=True)
        device = TextConverter(rsrcmgr, retstr, laparams=layout)

        fp = open(pdf, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos=set()

        for pg in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,
                                      caching=caching, check_extractable=True):
            interpreter.process_page(pg)

        text = retstr.getvalue()

        fp.close()
        device.close()
        retstr.close()
        return text 
    

    def save_file(self, filename, pdffile='', txt=''):
        """ save contract text to filename
    
        Parameters:
        filename (str) : filename to save to
        pdffile (str) : optional contract pdf file
        txt (str) : optional contract txt
        Returns: None
        
        """
       
        # no text
        if len(txt) == 0:
            if len(pdffile) > 0:
                txt = self.convert_pdf_to_txt(pdffile)  #get text from pdffile
            else:
                txt = self.text #objects text
        
        f = open(filename,'w')
        f.write(txt)
        f.close()
        
    
    def save_to_file(self, filename, txt):
        """ save txt to filename
    
        Parameters:
        filename (str) : filename to save to
        txt (str) : text to save
        Returns: None
        
        """
        f=open(filename,'w')
        f.write(txt)
        f.close()
        
        
   
    def save_pdf_to_txt(self, pdffile, txtfile):
        """ save pdf file to text 
    
        Parameters:
        pdffile (str) : pdf contract file
        txtfile (str) : filename to save to
        Returns: None
        
        """
        
        txt = self.convert_pdf_to_txt(pdffile)
        self.save_to_file(txtfile, txt)
        
        
    
    def get_doc(self, txt):
        """ returns NLP doc from text 
    
        Parameters:
        txt (str) : text to process
        Returns: spacy NLP doc from text
        
        """
        doc = self.nlp(txt)
        return doc
    
    
    # tags: https://spacy.io/api/annotation 
    def get_entities(self, doc):
        """ returns a dataframe of Named Entities 
    
        Parameters:
        doc : spacy doc 
        Returns: dataframe of Named Entities 
        
        """
        
        # for each named-entity in doc
        df = pd.DataFrame(
            [ent.text, ent.start_char, ent.end_char, ent.label_] for ent in doc.ents 
        )
        df.columns = ['text', 'start_char', 'end_char', 'label_']
        return df
    
    
    def print_entities(self, doc):
        """ print Named Entities 
    
        Parameters:
        doc : spacy doc 
        Returns: None 
        
        """
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
    
    def pretty_entities(self, doc):
        """ pretty print Named Entities 
    
        Parameters:
        doc : spacy doc 
        Returns: None 
        
        """
        pprint([(e.text, e.label_) for e in doc.ents])
        
    
    # Part-of-speech tags and dependencies
    # https://spacy.io/usage/linguistic-features#dependency-parse
    def get_tokens(self, doc):
        '''
        Text: original word text.
        Lemma: base form of the word.
        POS: simple part-of-speech tag.
        Tag: detailed part-of-speech tag.
        Dep: Syntactic dependency, i.e. the relation between tokens.
        Shape: word shape – capitalization, punctuation, digits.
        ''' 
        
        # Create list of word tokens, remove line feeds
        #for each token — i.e. a word, punctuation symbol, whitespace, etc.
        for token in doc:
           tokens = [t for t in doc if t.pos_ != 'SPACE']  #warning! takes too long for long txt
           
         
    
        df = pd.DataFrame(
            [t.text, t.lemma_, t.pos_, t.tag_, t.dep_,
              t.shape_, t.is_alpha, t.is_stop, t.is_title] for t in tokens
        )
        df.columns = ['text', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop', 'is_title']
        return df
    
    
    def get_noun_chunks (self, doc):
        """ returns noun chucks from document
    
        Parameters:
        doc : spacy doc
        Returns: base noun phrases in the document.
        
        """
        noun_chunks = list(doc.noun_chunks) # base noun phrases in the document.
        #for token in noun_chunks:
        #    nouns = [t for t in noun_chunks if t.pos_ != 'SPACE'] 
        return noun_chunks
    
    
    def get_pos_list (self, doc, pos="NOUN"):
        """ returns a list of tokens matching parts of speech 
    
        Parameters:
        doc : spacy doc 
        pos (str): parts-of-speech tag. eg: "NOUN", "VERB", etc
        Returns: None 
        
        """
        
        return [token.text for token in doc if token.pos_ == pos] 
       
        
   
    def sent_segment(self, txt):
        """ sentence tokenization
    
        Parameters:
        txt : tex to tokenize into sentences
        Returns: list of sentences
        
        """

        # Load English tokenizer, tagger, parser, NER and word vectors
        nlp = English() 

        # A simple pipeline component, to allow custom sentence boundary detection logic 
        # that doesn’t require the dependency parse. It splits on punctuation by default
        sbd = nlp.create_pipe('sentencizer')

        # Add the component to the pipeline
        nlp.add_pipe(sbd)

        #nlp is used to create documents with linguistic annotations.
        doc = nlp(txt)   

        # create list of sentence tokens
        sents_list = []
        for sent in doc.sents:
            sents_list.append(sent.text)
            
        return sents_list 
    


    # Visualizing the dependency parse
    # https://spacy.io/usage/visualizers
    # https://spacy.io/api/top-level#displacy_options

    
    #dependency visualizer, dep, shows part-of-speech tags and syntactic dependencies.
    def viz_deps(self, doc, dispoptions=DISPLAY_OPTIONS):
        displacy.render(doc, style="dep", jupyter=True, options=dispoptions)
        
    # Visualizing long texts    
    def viz_deps_long(self, doc, dispoptions=DISPLAY_OPTIONS):
        sentence_spans = list(doc.sents)
        sentence_spans
        displacy.render(sentence_spans, style="dep", jupyter=True, options=dispoptions)

    # Visualizing the entity recognizer
    def viz_ent(self, doc, dispoptions=DISPLAY_OPTIONS):
        displacy.render(doc, style="ent", jupyter=True, options=dispoptions)
        
        
        


from pathlib import Path

def getTxtFileName(pdffile, ext):
    return os.path.splitext(pdffile)[0]+ ext

def getFilePath(file):
    path = Path(file).parent.absolute()
    return path



def test(pdf, savejson=False, savetxt=False):

    txtfile =  getTxtFileName(pdf, '.txt') 

    con = Contract(
                pdffile=pdf
                #txtfile=txtfile
                )

    # write to text file
    if savetxt:
        con.save_pdf_to_txt(pdf, txtfile)


    # test json
    con.getJson(savejson) 
    con.printJson(con.contract)
    return con


    # test
    filenames  = []
    parent_dir = os.getcwd()+'/data'
    for pdf_file in glob.glob(os.path.join(parent_dir, '*.pdf')):
        filenames.append(pdf_file)

    contracts = []
    for fn in filenames:
        print(fn)
        c=test(pdf=fn, savejson=True, savetxt=False)
        contracts.append(c)









import pandas as pd
from nltk.corpus import wordnet
from itertools import chain


def createSearch(search):
    d = eval(search) 
    data = pd.DataFrame(d.items(), columns=['Category', 'Keyword'])
    return data


#find synonyms based on the keyword and build into a dataframe
def get_synonyms(df, col, num):
    L = []
    for key in df[col]:
        syn = wordnet.synsets(key)
       
        #flatten all lists by chain, remove duplicates by set
        lemmas = list(set(chain.from_iterable([w.lemma_names() for w in syn])))
        
        for syn in lemmas[:num]:
            L.append([key, syn])
    
    return (pd.DataFrame(L, columns=['Keyword','Syn']))    


def doSearch(contracts, searchStr):
    
    data = createSearch(searchStr)

    # add dups
    #data.loc[3] = ['financial', 'escrow']
    #data.loc[4] = ['SLA', 'breach']
    #data
    
    #add number of filtered synonyms
    df1 = get_synonyms(data, 'Keyword', 3)  # up to 3 syn
    #print(df1)

    synonyms = list(df1['Syn'])
    #print(synonyms)

    #merge Category, Keyword and Synonyms 
    df2= pd.merge(data, df1)
    #print(df2)

    # merge keywords and syn
    category=list(data['Category'])
    keyword=list(data['Keyword'])

    new_input_data= keyword + synonyms
    new_input_data = list(set(new_input_data))
    #print(new_input_data)

    #clean Synonyms without any duplication
    resultwords = [word for word in new_input_data if word not in keyword]
    #print(resultwords)
    
    
    ##### do the search
    colnames = ('filename', 'section', 'category', 'keyword_synonym', 'original_keyword', 'context')
    foundDf = pd.DataFrame(columns=colnames)

    # search in each contract file
    for con in contracts:
        
        for sec in con.sections:
            
            seclist = sec.split('\n')
            secType = seclist[0]
            stext =''.join(seclist)
            
            reccnt=0
            
            for word in new_input_data:
                if word in keyword:
                    match= data.loc[data['Keyword'] == word, 'Category'].iloc[0] 
                else: 
                    match= df2.loc[df2['Syn'] == word, 'Category'].iloc[0]           
                if word in resultwords:
                    match2 = df2.loc[df2['Syn'] == word, 'Keyword'].iloc[0]
                else: 
                    match2=""

                sub = '(?:\S+\s+){10}%s.*(?:\S+\s+){10}' % word
                surrounding_text = re.findall(sub, stext.lower()) 

                if surrounding_text != [] :
                    
                    for found_text in surrounding_text:
                        found_text = found_text.replace('\n','').replace(',',' ')
                        
                        #print(secType, match, word, match2, cnt)
                            
                        foundDf.loc[reccnt] = [con.pdffile, secType, # filename', 'section_num
                                            match, word,             # 'category', 'Keyword_Synonym'
                                            match2, found_text]      # Original_Keyword', 'Context
                       
                
                reccnt = reccnt+1
                


    # save results to csv
    if len(foundDf) > 0:
        path= os.getcwd()+'/data/foundList.xlsx'
        print(path)
        foundDf.to_excel(path, index = False)
        print('Check foundList.xlsx')
        
    return foundDf
        
        
