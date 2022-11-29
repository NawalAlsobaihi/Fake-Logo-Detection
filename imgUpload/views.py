from django.shortcuts import render
from .forms import ImageUploadForm

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json



# Model saved with Keras model.save()
MODEL_PATH ='/home/nawal/Dev/playground/Tenserflowv2/model_resnet50.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def decode_predictions(preds, top=4, class_list_path='/home/nawal/Dev/playground/Tenserflowv2/flickerlogo27_index.json'):
 
  if len(preds.shape) != 2 or preds.shape[1] != 27: # your classes number
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
                      
  index_list = json.load(open(class_list_path))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(index_list[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results
















# Create your views here.

def handle_uploaded_file(f):
    with open("img.jpg",'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def home(request):
    return render(request,'home.html')

def imageprocess(request):
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])
        

        img_path = 'img.jpg'
        
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = model.predict(x)
        print('Predicted:',decode_predictions(preds, top=3)[0])
        
        
        #to display in Frontend
        html = decode_predictions(preds, top=3)[0]
        res = []
        for e in html:
            res.append((e[1],np.round(e[2]*100,2)))
            
        if (res[0][1] >= 95 and res[0][1] <= 100):
           predect_res='Orignal'
        else:
           predect_res='Fake'
           
        return render(request,'result.html',{'res':res ,'result':predect_res})
        
        
       
        #return render(request,'result.html',{'result':preds})

    return render(request,'result.html')
    


