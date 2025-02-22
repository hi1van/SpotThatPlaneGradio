import gradio as gr
from fastai.vision.all import *

learn = load_learner('model.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Spot that Plane"
description = "A plane model classifier trained with fastai. Created as a demo for Gradio and HuggingFace Spaces."
examples = []

gr.Interface(fn=predict,inputs=gr.Image(type="pil"),outputs=gr.Label(num_top_classes=3),title=title,description=description,examples=examples).launch()