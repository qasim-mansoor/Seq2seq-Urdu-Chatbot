import yaml
from yaml.loader import SafeLoader
import re
import numpy as np
import pandas as pd
import os
### 1. EXTRACT THE DATA ; SEPARATE THE QUESTIONS AND ANSWERS

# Open the file and load the file

dir_path = 'ConvDataset/'
files_list = [f for f in os.listdir(dir_path) if f.endswith('.yml')]

q_a = []
for name in files_list:
    with open(dir_path + name, encoding='utf-8') as f:
        dataset = yaml.load(f, Loader=SafeLoader)
        pairs = dataset['conversations']  # ex: pairs[0] represents a conversation
        q_a.append(pairs)


questions = list()
answers = list()

for conv in q_a:
    if len(conv) == 2 :
        questions.append(conv[0])
        answers.append(conv[1])
    elif len(conv) > 2 :
        for i in range(len(conv)-1):
            questions.append(conv[0])
            answers.append(conv[i+1])

# In the answers list there are some answers which are dictionaries, not strings.'
i=0
while i< len(answers):
    if type(answers[i]) == dict:
        answers.pop(i)
        questions.pop(i)
        i -= 1
    i +=1

### 2. PREPROCESSING THE DATA

# Clean the dataset

def clean_data(sentence):
    sentence = sentence.lower()

    sentence = re.sub(",", " , ", sentence)
    sentence = re.sub("\.", " . ", sentence)
    sentence = re.sub("\|", " | ", sentence)
    sentence = re.sub("-", " - ", sentence)
    sentence = re.sub("\?", " ? ", sentence)
    sentence = re.sub("!", " ! ", sentence)
    sentence = re.sub("\"", " \" ", sentence)
    sentence = re.sub("\'", " ' ", sentence)
    sentence = re.sub("\(", " ( ", sentence)
    sentence = re.sub("\)", " ) ", sentence)
    sentence = re.sub("\{", " { ", sentence)
    sentence = re.sub("\{", " } ", sentence)
    sentence = re.sub("\<", " < ", sentence)
    sentence = re.sub("\>", " > ", sentence)
    sentence = re.sub("\;", " ; ", sentence)
    sentence = re.sub("\:", " : ", sentence)
    sentence = re.sub(u"\u0964", " "+u"\u0964"+" ", sentence)
    sentence = re.sub(u"\u0965", " "+u"\u0965"+" ", sentence)
    sentence = re.sub(u"\u09F7", " "+u"\u09F7"+" ", sentence)
    sentence = re.sub(u"\u09FB", " "+u"\u09FB"+" ", sentence)
    sentence = re.sub("\s+", " ", sentence)
    sentence = re.sub("&amp ;", "&amp;", sentence)
    sentence = re.sub("&quot ;", "&quot;", sentence)
    sentence = re.sub("&quote ;", "&quote;", sentence)
    # remove all the punctuation
    sentence = re.sub(r"[\؟,.?:;_'!()\"-]", "", sentence)
    return sentence

clean_questions = []
clean_answers = []

for dataset in questions:
    clean_questions.append(clean_data(dataset))

for dataset in answers:
    clean_answers.append(clean_data(dataset))

# Need to add tags for ending (eoa) and starting (boa) of sentences for decoders

START = "boa "
END = " eoa"

decoder_inputs = []

for answer in clean_answers:
    dec_answer = START + answer + END
    decoder_inputs.append(dec_answer)

encoder_inputs = clean_questions

# Storing the encoder and deocder inputs as numpy arrays for further use
encoder_inputs_np = np.array(encoder_inputs)
np.save('ConvDataset_utils/encoder_inputs', encoder_inputs_np)
pd.DataFrame(encoder_inputs_np).to_csv('ConvDataset_utils/encoder_inputs.csv')

decoder_inputs_np = np.array(decoder_inputs)
np.save('ConvDataset_utils/decoder_inputs', decoder_inputs_np)
pd.DataFrame(decoder_inputs_np).to_csv('ConvDataset_utils/decoder_inputs.csv')

