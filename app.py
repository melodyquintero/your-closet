import os
from flask import Flask, request, jsonify

import keras
from keras.preprocessing import image
from keras import backend as K

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'

model = None
graph = None

# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html


def load_model():
    global model
    global graph
    model = keras.models.load_model("fashionmnist_trained.h5")
    graph = K.get_session().graph


load_model()


def prepare_image(img):
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Scale from 0 to 255
    img /= 255
    # Invert the pixels
    img = 1 - img
    # Flatten the image to an array of pixels
    image_array = img.flatten().reshape(-1, 28 * 28)
    # Return the processed feature array
    return image_array


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)

            # Load the saved image using Keras and resize it to the mnist format of 28x28 pixels
            image_size = (28, 28)
            im = image.load_img(filepath, target_size=image_size, grayscale=True)

            # Convert the 2D image to an array of pixel values
            image_array = prepare_image(im)
            print(image_array)

            # Get the tensorflow default graph and use it to make predictions
            global graph
            with graph.as_default():

                # Use the model to make a prediction
                predicted_digit = model.predict_classes(image_array)[0]
                labels=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
                prediction = labels[predicted_digit]
                data["prediction"] = prediction

                # indicate that the request was a success
                data["success"] = True


                return jsonify(data)
            
    return '''

    
    <!doctype html>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
    
    <body background="https://furnitureinredsea.com/wp-content/uploads/2018/07/closets-closet-organizers-california-closets-walk-in-closet-concerning-california-closet-laundry-room-of-california-closet-laundry-room.jpg">
    
    <title>Your Digital Closet</title>

    <style>
    .container {
    opacity: 0.85;
    background: white;
    color:black;
    border-radius: 25px;
    position: absolute;
        top: 30%;
        left: 50%;
    overflow-y: scroll;
    padding: 30px;
    }
    </style>

    
    <div class="container">

    <h1>Your Digital Closet</h1>
    
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Classify Your Item>
        </p>
    </form>

    </div>

    

   

    '''


if __name__ == "__main__":
    app.run()
