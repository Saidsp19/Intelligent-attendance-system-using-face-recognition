from keras_facenet import FaceNet
from pil import Image
from matplotlib import pyplot
from numpy import asarray
from os import listdir
import re
from sklearn import preprocessing
import numpy as np

embedder = FaceNet()
#face landmarks using multitask cnn
#detections = embedder.extract('mahesh (9).jpg', threshold=0.95)


def show_face(filename,box,required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
	# convert to RGB, if needed
    image = image.convert('RGB')
	# convert to array
    pixels = asarray(image)
    
    x1,y1,width,height = box
    x2, y2 = x1 + width, y1 + height
        
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
#    image = image.resize(required_size)
    face_array = asarray(image)
    pyplot.imshow(face_array)
    pyplot.show()
    return
                          
    




# specify folder to plot
folder = 'images//'
i = 0
face_encoding = []
face_names = []
face_information = []
paths = []
face_box = []
# enumerate files
for filename in listdir(folder):
	# path
    path = folder + filename
    paths.append(path)
    print(i)
    i = i+1
	# get face
    detections = embedder.extract(path, threshold=0.95)
#    detection -  0,1,2
    face_info = detections[0]
    face_information.append(face_info)
    box = face_info['box']
    face_box.append(box)
    # show_face(path,box)
    
    face_embedding = face_info['embedding']
    face_encoding.append(face_embedding)
    face_names.append(re.sub("[0-9,(,),'']",'',filename[:-4]))
    

label_encoder = preprocessing.LabelEncoder()
face_names_labels = label_encoder.fit_transform(face_names)

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
names = unique(face_names)
#labels = unique(face_names_labels)


np.save('face_enoding',face_encoding)
np.save('face_names_labels',face_names_labels)
np.save('names',names)