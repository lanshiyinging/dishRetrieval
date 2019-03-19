from flask import *
import tensorflow as tf
from network import get_hashcode
from network import retrieval
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/service', methods=['GET', 'POST'])
def service():
    upload_path = ''
    result={
        "rank1": "",
        "rank2": "",
        "rank3": "",
        "rank4": "",
        "rank5": ""
    }
    if request.method == 'POST':
        f = request.files['file']
        upload_path = 'static/' + secure_filename(f.filename)
        f.save(upload_path)
        query_hash_code = get_hashcode.get_hashcode(upload_path)
        result_ = retrieval.retrieval(query_hash_code)
        result["rank1"] = result_[0][0]
        result["rank2"] = result_[1][0]
        result["rank3"] = result_[2][0]
        result["rank4"] = result_[3][0]
        result["rank5"] = result_[4][0]
        return redirect(url_for('service'))
    return render_template('index.html', input_image_path=upload_path, result=result)

if __name__ == '__main__':
    app.run(debug=True)


