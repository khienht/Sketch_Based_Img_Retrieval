from flask import Flask, render_template, request
import json
import os
import time
import base64
import torch
from datetime import timedelta
from models.vgg import vgg16
from torch import nn
from utils.retrieval_demo import Retrieval

net_dict_path = 'model/photo_vgg16_8111.pth'
'''vgg'''
vgg = vgg16(pretrained=False)
vgg.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
vgg.load_state_dict(torch.load(net_dict_path, map_location=torch.device('cpu')))

# for retrieval
extract = Retrieval(vgg)

# SketchAPP definition
app = Flask(__name__, template_folder='templates', static_folder='static')
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/canvas', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        sketch_src = request.form.get("sketchUpload")
        upload_flag = request.form.get("uploadFlag")
        sketch_src_2 = None
        if upload_flag:
            sketch_src_2 = request.files["uploadSketch"]
        if sketch_src:
            flag = 1
        elif sketch_src_2:
            flag = 2
        else:
            return render_template('canvas.html')

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/sketch_tmp', 'upload.png')

        if flag == 1:
            # base64 image decode
            sketch = base64.b64decode(sketch_src[22:])
            user_input = request.form.get("name")
            file = open(upload_path, "wb")
            file.write(sketch)
            file.close()

        elif flag == 2:
            # upload sketch
            sketch_src_2.save(upload_path)
            user_input = request.form.get("name")

        retrieval_list, real_path = extract.retrieval(upload_path)
        real_path = json.dumps(real_path)

        return render_template('panel.html', userinput=user_input, val1=time.time(), 
                               upload_src=sketch_src,
                               retrieval_list=retrieval_list,
                               json_info=real_path)

    return render_template('canvas.html')


@app.route('/canvas')
def homepage():
    return render_template('canvas.html')

if __name__ == '__main__':
    # open debug mode
    app.run(debug=True)