from flask import jsonify
from app import app, auth

@app.route('/')
def home():
    return "Funcionando"