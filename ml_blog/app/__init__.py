from flask import Flask

app = Flask(__name__)

app.config.from_object("config")

from app import views

__all__ = []