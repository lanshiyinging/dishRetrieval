from flask import *
import tensorflow as tf
#from network import get_hashcode_v2
#from network import retrieval_v2
import os
import sys
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap

sys.path.append('../../network')
import get_hashcode_v2_web
import retrieval_v2_web

app = Flask(__name__)
bootstrap = Bootstrap(app)
root_path = "/root/lsy/dishRetrieval/"
#root_path = "/Users/lansy/Desktop/graduateDesign/dishRetrieval/"

@app.route('/index')
def index():
    return render_template('index.html')

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
        #upload_path = '/root/lsy/dishRetrieval/web/app/static/' + secure_filename(f.filename)
        #upload_path = root_path + 'web/app/static/' + secure_filename(f.filename)
        upload_path = 'static/' + secure_filename(f.filename)
        f.save(upload_path)
        query_hash_code = get_hashcode_v2_web.get_hashcode(upload_path)
        result_ = retrieval_v2_web.retrieval(query_hash_code)
        result["rank1"] = result_[0][0].replace('../data', '../static')
        result["rank2"] = result_[1][0].replace('../data', '../static')
        result["rank3"] = result_[2][0].replace('../data', '../static')
        result["rank4"] = result_[3][0].replace('../data', '../static')
        result["rank5"] = result_[4][0].replace('../data', '../static')
        print(result)
        #return redirect(url_for('service'))
    return render_template('index.html', input_image_path='../'+upload_path, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


