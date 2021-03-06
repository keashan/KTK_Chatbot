from flask import Flask,request,jsonify,render_template
from flask_cors import CORS
from torch_utils import chat_response
from model import NuralNet
import sys


app = Flask(__name__)
CORS(app)

@app.route('/chat',methods=['POST'])
def chat():
    if request.method=='POST':
        sentence=request.args.get('sentence')
        if sentence is None or sentence=='':
            return jsonify({'error':'no message'})
        try:
            message=chat_response(sentence)
            return jsonify({'KTK_Bot' : message})
        
        except:
            return jsonify({'error': 'error in replying'})
    
    #return jsonify({'result':1})

@app.route('/')
def homepage():
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)