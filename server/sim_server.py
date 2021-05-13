from flask import Flask, request, json, jsonify
import os, sys

from flask.helpers import make_response
sys.path.append("..")
import time
from flask_cors import * 
from run_predict import  load_model, predict

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 设置跨域
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
model = load_model()

@app.route("/", methods=["GET"])
def init():
    return jsonify({'info':'文本摘要API'})

@app.route("/summary", methods=["GET","POST"])
def summary():
    text = None
    if request.method == "POST":
        text = request.json.get("text")
    if request.method == "GET":
        text = request.args.get("text", type=str)
    if text is None or text.__len__() == 0:
        return jsonify({'res':'Text不能为空'})
    
    # 开始预测
    res = predict(model, text)
    res = res.replace(" ","")
    response = make_response(jsonify({'res':res}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__=="__main__":
    app.run(port=5555, debug=True)