# -*- coding: utf-8 -*-
"""This is where we are deploying the server"""
import os
import sys
import click
from app import app
from model.evaluate import Evaluate
from model.tokenizer import FlikerTokenizer


def except_hook(cls, exception, traceback):
    """Give us back the original exception hook that may have been changed by Qt"""
    sys.__excepthook__(cls, exception, traceback)


def resource_path(relative_path):
    """Get absolute path to a resource. Works for development and for releases with PyInstaller"""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


@click.command()
@click.argument("files", nargs=-1)
def main(files=None):
    """This is server creates joke for uploaded image"""
    sys.excepthook = except_hook

    app.secret_key = "secretkey"
    fktok = FlikerTokenizer()
    app.config["TOKENIZER"] = fktok
    app.config["MODEL"] = Evaluate(fktok.tokenizer, fktok.max_length)
    app.config["UPLOAD_FOLDER"] = os.path.abspath("./uploads")
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024 * 6  # 6MB
    app.run(debug=True)


if __name__ == "__main__":
    main()
