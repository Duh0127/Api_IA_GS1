import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ESTÁ OCORRENDO UM ERRO NA COMPATIBILIDADE DE VERSÕES ENTRE O TENSORFLOW, KERAS E O NUMPY
# Apenas consegui rodar o meu projeto com as versões mais antigas dessas bibliotecas (tensorflow==2.12.0, numpy==1.23.5)

# Caso for rodar no replit, abrir o SHELL e digitar:  [ pip install tensorflow==2.12.0 ]  antes de executar o código

print("Versão do TensorFlow:", tf.__version__)
print("Versao do numpy: ", np.__version__)

# Diretório onde as imagens serão salvas
UPLOAD_FOLDER = "predictions"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Extensões permitidas
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Carregar o modelo Keras (.h5)
model = tf.keras.models.load_model("keras_model.h5")

# Carregar os rótulos
with open("labels.txt", "r", encoding="utf-8") as file:
    labels = file.read().splitlines()


# Função para pré-processamento de imagem
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array


# Função para fazer predições
def predict(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    return labels[predicted_class], confidence


# função para verificar se a extensão do arquivo é permitida
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello_world():
    return "Hello, World!"


# rota para salvar a imagem enviada no formato FORMDATA com o nome 'file'
@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "Arquivo não enviado"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nenhum arquivo encontrado"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Fazer a predição
        prediction, confidence = predict(filepath)

        # Excluir a imagem após a predição
        os.remove(filepath)

        return jsonify({"prediction": prediction, "confidence": float(confidence)}), 200
    else:
        return jsonify({"error": "Tipo da imagem não permitida"}), 400


# rota para puxar a imagem salva, passando apenas o nome do arquivo como parametro da rota
#  Exemplo: http://127.0.0.1:5000/get-image/nomedaimagem.jpg
@app.route("/get-image/<filename>", methods=["GET"])
def get_image(filename):
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except FileNotFoundError:
        return jsonify({"error": "Arquivo não encontrado"}), 404


if __name__ == "__main__":
    # Cria o diretório de upload se não existir
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host="0.0.0.0", port=8080)
