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

### test the code via flask web app
navigate to the test folder <br>
Create a directory called data at the same level as the python notebook or script.
Drop the contract files in data (note: currently only pdf is supported).
<b>start the test web app:</b>  python server.py <br>
<b>open this url in web browser:</b>  http://127.0.0.1:5000/ <br>


 
