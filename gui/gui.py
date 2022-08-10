import gradio as gr
from PIL import Image as p_image
from tensorflow.keras.preprocessing import image as imager
import numpy as np

class GUI:
    model = None
    def predict_image(img):
        img_width, img_hight = 224, 224
        img = p_image.fromarray(img)
        img = imager.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        prediction = __class__.model.predict(img)[0][0]
        
        print("Here  :  ", prediction)
        if prediction == 0.0:
            return 'Covid'
        else:
            return 'Normal'


    def web():
        image = gr.inputs.Image(shape=(224,224))
        label = gr.outputs.Label(num_top_classes=1)
        gr.Interface(fn=__class__.predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')