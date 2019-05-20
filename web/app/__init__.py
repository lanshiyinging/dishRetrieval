from flask import Flask, render_template
from flask_bootstrap import Bootstrap


bootstrap = Bootstrap()


def create_app(config_name):
    app = Flask(__name__)
    bootstrap.init_app(app)

    return app