import os

from flask import request
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
            try:
                model = app.config["MODEL"]
                filename = app.config["UPLOAD_FOLDER"] + "/" + filename
                resp = model.evaluate(filename)
                resp = resp[:-1]
                if resp:
                    resp = {"message": ' '.join(resp)}, 200
                else:
                    resp = {"message": "Sorry!! Unable to crack jokes now."}, 400
            except Exception as e:
                resp = {"message": "Unable to crack jokes now!! ", error: str(e)}, 500
            os.remove(filename)
            return resp
        else:
            resp = {"message": "Allowed file types are %s" % ALLOWED_EXTENSIONS}, 400
            return resp


api.add_resource(JokePlease, "/crack-joke")
