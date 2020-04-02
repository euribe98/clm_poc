# metricstream_poc
## Install dependencies
pip install -r requirements.txt


## Download best-matching version of specific model for your spaCy installation
python -m spacy download en_core_web_sm

## Out-of-the-box: download best-matching default model and create shortcut link
python -m spacy download en


## test the model
import spacy<br>
nlp = spacy.load("en_core_web_sm")

 
