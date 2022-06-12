from flask import send_from_directory
from flask_restful import Resource
from werkzeug.utils import secure_filename

from app import api, app


class FileServe(Resource):
    def get(self, filename):
        try:
            filename = secure_filename(filename)
            # filename = "%s/%s" % (app.config["UPLOAD_FOLDER"], filename)
            return send_from_directory(app.config["UPLOAD_FOLDER"], path=filename, as_attachment=True)
        except FileNotFoundError:
            return "File not found.", 404


api.add_resource(FileServe, "/file/<string:filename>")
