from flask import Flask
from flask import request
from flask_cors import CORS
import pickle
from analysis import remove_noise
from nltk.tokenize import word_tokenize


def classify(message):
    f = open('sentiment_classifier', 'rb')
    classifier = pickle.load(f)
    f.close()

    custom_message = message

    custom_tokens = remove_noise(word_tokenize(custom_message))

    result = classifier.classify(dict([token, True] for token in custom_tokens))

    print(custom_message)

    return result


app = Flask(__name__)

CORS(app)


@app.route("/classify", methods=['GET'])
def get_classify():
    return classify(request.args.get('message'))


if __name__ == "__main__":
    app.run()
