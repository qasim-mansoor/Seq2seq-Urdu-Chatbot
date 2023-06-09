# Seq2seq-Urdu-Chatbot
A Chatbot system made using seq2seq2 architecture with LSTM-based encoder-decoders.


# Word Embeddings
The word embeddings used to train this model can be found at https://github.com/samarh/urduvec
The word embeddings must be in in a txt file and be placed in the same directory as the model+Embed.py file

# Usage
To just run the app as is all you need to do is to open a terminal in the the current directory and type:
```
flask run
```

To train your own model on your own dataset, you must place your dataset in the ConvDataset directory, as a yml file following the template of the provided file. Secondly, you must train the bot using the files provided in the Prerequisites directory, the sequence is as follows:

```
preprocessingData.py > vocabulary.py > model+Embed.py 
```

You can replace the model+Embed.py file with model.py file if you would like to use a model without any embedding.

Finally, to run the chatbot you must use the inferenceModel_embedded.py file (or the inferenceModel.py file if no embeddings were used).
