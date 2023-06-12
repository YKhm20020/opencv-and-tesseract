import pathlib
import urllib.request
import cv2
from cv2 import dnn_superres  # $ pip install opencv-contrib-python
import pytesseract

def load_sr_model(model_name, scale, url):
    model_path = pathlib.Path("models").joinpath("{}_{}.pb".format(model_name, scale))
    if not model_path.exists():
        model_path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, model_path)

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(model_name, scale)
    return sr

img = cv2.imread('./sample.png')
sr = load_sr_model("lapsrn", 4, "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb")
img_sr = sr.upsample(img)
result = pytesseract.image_to_string(img, lang="jpn", config="--psm 6")