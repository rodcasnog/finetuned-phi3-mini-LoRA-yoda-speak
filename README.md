# finetuned-phi3-mini-LoRA-yoda-speak
Finetuned Phi 3 mini 4k that translates short English sentences into Yoda speak!

To use it yourself, first install the requirement packages from the command line (preferably from a virtual environment):
```
pip install requirements.txt
```
Next, to train it yourself simply follow the trainig code at `train.ipynb` or generate your own translations at `inference.ipynb`. You may also generate your translations directly from the comand line by typing:
```
python translate.py 'Your phrase here will be translated.'
```
You get:
```
Input: Your phrase here will be translated.
Translation: Translated, your phrase here will be.
```
