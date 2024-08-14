from flask import Flask, request, render_template
import io
import torchvision.transforms as transforms
import torch
from torchvision import models
from PIL import Image


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

model = torch.load("sarvesh_legume_net.pth", map_location='cpu')
model.eval()


class_names = ['chana_dal_split_chickpea',
 'masoor_split_red_lentils',
 'moong_mung_bean',
 'muth_moth_bean',
 'urad_black_gram']


def transform(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transform = transforms.Compose([
        transforms.Resize(256),  # Resize first
        transforms.CenterCrop(224),  # Then crop
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # Normalize the tensor
    ])
    return data_transform(image).unsqueeze(0)



@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image file
    image = request.files['image']
    
    # Process image and make prediction
    image_tensor = transform(Image.open(image))
    output = model(image_tensor)

    # Get class probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    probabilities = probabilities.detach().numpy()[0]

    # Get the index of the highest probability
    class_index = probabilities.argmax()

    # Get the predicted class and probability
    predicted_class = class_names[class_index]
    probability = probabilities[class_index]

    # Sort class probabilities in descending order
    class_probs = list(zip(class_names, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)

    # Render HTML page with prediction results
    return render_template('predict.html', class_probs=class_probs,
                           predicted_class=predicted_class, probability=probability)



"""
with open("static/img/IMG_1390.jpg", 'rb') as f:
    image_bytes = f.read()
    tensor = transform(image_bytes=image_bytes)
    print(tensor)
"""
    

def get_prediction(image_bytes):
    tensor = transform(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat

if __name__ =="__main__":
    app.run(debug=True)


