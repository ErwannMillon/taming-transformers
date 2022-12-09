import bz2
import os
import os.path as osp
import sys
from multiprocessing import Pool
import dlib
import numpy as np
import PIL.Image
import requests
import scipy.ndimage
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import gradio as gr
from edit import blend_paths
from loaders import load_default
LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

model = load_default()
with gr.Blocks() as demo:
    gr.Markdown("""
    **Income Classification with XGBoost 💰**:  This demo uses an XGBoost classifier predicts income based on demographic factors, along with Shapley value-based *explanations*. The [source code for this Gradio demo is here](https://huggingface.co/spaces/gradio/xgboost-income-prediction-with-explainability/blob/main/app.py).
    """)
    with gr.Row():
        with gr.Column():
            lip_size = gr.Slider(
                label="lip size",
                minimum=-1.5,
                maximum=1.9,
                step=0.1,
            )
            blend_weight = gr.Slider(
                label="0 is src image, 1 is blend_img",
                minimum=-0.,
                maximum=1.,
                step=0.1,
            )
            base_img = gr.Image(label="base Image", type="filepath")
            blend_img = gr.Image(label="image for face blending (optional)", type="filepath")

        with gr.Column():
            out = gr.Image(interactive=False)
            # blend_img.change(blend_paths, inputs=[model, base_img, blend_img, {"weight": blend_weight}],
                            # outputs=[out])
            # blend_weight.change(blend_paths, inputs=[model, base_img, blend_img, {"weight": blend_weight}],
                                # outputs=[out])
            # lip_size.change()
            # base_img = gr.Image(interactive=False)
            # interpret_btn.click(
            #     interpret,
            #     inputs=[
            #         age,
            #         work_class,
            #         education,
            #         years,
            #         marital_status,
            #         occupation,
            #         relationship,
            #         sex,
            #         capital_gain,
            #         capital_loss,
            #         hours_per_week,
            #         country,
            #     ],
                # outputs=[plot],
            # )
            i = gr.Button()
            i.click(blend_paths, inputs=[model, base_img, blend_img])

        base_img.change(lambda x:x, inputs=base_img, outputs=out)
        

demo.launch()