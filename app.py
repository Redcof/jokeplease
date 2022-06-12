from flask import Flask, render_template
from flask_restful import Api

app = Flask(__name__)
api = Api(app)


@app.route("/")
def hello_world():
    return render_template('index.html')


# imports to include the modules
from modules import file_serve, joke_please
