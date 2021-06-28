
import json
import logging

from flask import Flask, request, jsonify
from agent import Service

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_service = Service()

@app.route('/')
def home():
	return "IITP fact reasoning App Works!"

@app.route('/fact_reasoning/task_list', methods=['GET'])
def task_list():
	return Service.get_task_list()

@app.route('/fact_reasoning', methods=['POST'])
def do_task():
	content = request.json
	return api_service.do(content)

