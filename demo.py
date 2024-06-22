from flask import Flask, render_template, request, session
import json
import os
import time
import base64
import torch
from datetime import timedelta
from models.vgg import vgg16
from models.sketch_resnet import resnet50
from torch import nn
from utils.retrieval_demo import Retrieval

net_dict_path = 'model/vgg16/photo_vgg16_29.pth'
PHOTO_RESNET = 'model/rn50_bs32_mg1_lr3_10class/photo_resnet50_20.pth'

def load_model(model_name):
    if model_name == 'vgg16':
        model = vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
        model.load_state_dict(torch.load(net_dict_path, map_location=torch.device('cpu')))
    elif model_name == 'resnet50':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(in_features=2048, out_features=10)
        model.load_state_dict(torch.load(PHOTO_RESNET, map_location=torch.device('cpu')))
    return Retrieval(model,model_name)

# SketchAPP definition
app = Flask(__name__, template_folder='templates', static_folder='static')
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/canvas', methods=['POST', 'GET'])
def upload():
    model_name = request.form.get("model_select", "resnet50")
    if request.method == 'POST':
        extract = load_model(model_name)
        sketch_src = request.form.get("sketchUpload")
        upload_flag = request.form.get("uploadFlag")
        
        sketch_src_2 = None
        if upload_flag :
            sketch_src_2 = request.files["uploadSketch"]
        
        print(sketch_src_2)
        if sketch_src:
            flag = 1
        elif sketch_src_2 :
            flag = 2
        else:
            return render_template('canvas.html')
        print(flag)
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/sketch_tmp', 'upload.png')
        if flag == 1:
            # base64 image decode
            sketch = base64.b64decode(sketch_src[22:])
            if sketch!= b'':
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
                                json_info=real_path, model_name=model_name)

    return render_template('canvas.html', model_name=model_name)
        
@app.route('/canvas')
def homepage():
    return render_template('canvas.html',model_name='resnet50')

if __name__ == '__main__':
    # open debug mode
    app.run(debug=True)