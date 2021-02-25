import random
import json
import torch
from model import NuralNet
from nltk_utils import  bag_of_words, tokenize

#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents=json.load(f)

FILE='data.pth'
data=torch.load(FILE)

input_size=data['input_size']
output_size=data['output_size']
hidden_size=data['hidden_size']
all_words=data['all_words']
tags=data['tags']
model_state=data['model_state']

model=NuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()


def chat_response(sentence):
    
    sentence=tokenize(sentence)
    X=bag_of_words(sentence,all_words)
    X=X.reshape(1,X.shape[0])
    X=torch.from_numpy(X)

    output=model(X)
    _, predicted = torch.max(output, dim=1)
    tag=tags[predicted.item()]

    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]
    if prob.item()>0.90:
        for intent in intents['intents']:
            if tag==intent['tag']:
                return random.choice(intent['responses'])
    elif prob.item()>0.80:
        for intent in intents['intents']:
            if tag==intent['tag']:
                return "Please ask something similar to f{random.choice(intent['patterns'])}"
    else:
        #return sentence
        return "I'm sorry, this is beyoned my knowledge. Please send an email to support@ ktktools.net."
