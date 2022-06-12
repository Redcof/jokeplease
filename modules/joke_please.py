import os

import PIL.Image as Image
from flask import request, url_for
from flask_restful import Resource
from werkzeug.utils import secure_filename

from app import api, app

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif", "svg"])


class JokePlease(Resource):
    """Read the barcode and process"""

    @staticmethod
    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def post(self):
        # check if the post request has the file part
        if "image" not in request.files:
            resp = {"message": 'No "image" in the request'}, 400
            return resp
        file = request.files["image"]
        if file.filename == "":
            resp = {"message": "No file selected for uploading"}, 400
            return resp
        if file and self.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            # read image for code
            model = app.config["MODEL"]
            resp = model.evaluate(filename)
            if resp[1] is False:
                resp = {"message": resp[0]}, 400
            elif resp[0]["code"] == "":
                resp = {"message": "No code was found"}, 400
            else:
                resp = {"message": resp[0], "url": url_for("fileserve", filename=filename)}, 200
            return resp
        else:
            resp = {"message": "Allowed file types are %s" % ALLOWED_EXTENSIONS}, 400
            return resp


api.add_resource(JokePlease, "/crack-joke")
