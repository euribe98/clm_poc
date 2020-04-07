# metricstream_poc
### Install dependencies
pip install -r requirements.txt

### Add pdf contract files
Create a directory called data at the same level as the python notebook or script. <br>
Drop the contract files in data (note: currently only pdf is supported).

### Download best-matching version of specific model for your spaCy installation
python -m spacy download en_core_web_sm

### Out-of-the-box: download best-matching default model and create shortcut link
python -m spacy download en


###  model load
import spacy<br>
nlp = spacy.load("en_core_web_sm")

 
