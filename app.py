from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the text classification models
with open('bilstm_model.pkl', 'rb') as f:
    bilstm_model = pickle.load(f)

with open('lstm_model.pkl', 'rb') as f:
    lstm_model = pickle.load(f)

with open('gru_model.pkl', 'rb') as f:
    gru_model = pickle.load(f)

# Load the classification tokenizer
with open('tctokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
   # Load the generation tokenizer
with open('tgtokenizer.pkl', 'rb') as f:
    tgtokenizer = pickle.load(f)
    
# Load the text generation models
with open("tgbilstm_model.pickle", "rb") as f:
    tgbilstm_model = pickle.load(f)

with open("tglstm_model.pickle", "rb") as f:
    tglstm_model = pickle.load(f)

with open("tggru_model.pickle", "rb") as f:
    tggru_model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify_text')
def classify_text():
    return render_template('classify.html')

@app.route('/generate_text')
def generate_text():
    return render_template('generate.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    
    max_length = 100
    padding_type = 'post'
    trunc_type = 'post'
    
    text = request.form['text']
    model = request.form['model']

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    if model == 'bilstm':
        prediction = bilstm_model.predict(padded_sequence)[0][0]
    elif model == 'lstm':
        prediction = lstm_model.predict(padded_sequence)[0][0]
    elif model == 'gru':
        prediction = gru_model.predict(padded_sequence)[0][0]

    probability = round(prediction * 100, 2)

    return render_template('result.html', text=text, model=model, probability=probability)

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        seed_text = request.form['seed_text']
        model_name = request.form['model']
        next_words = int(request.form['next_words'])

        # Select the appropriate loaded model based on the selected model
        if model_name == 'bilstm':
            model = tgbilstm_model
        elif model_name == 'lstm':
            model = tglstm_model
        elif model_name == 'gru':
            model = tggru_model
        else:
            return "Invalid model selection"

        # Generate text
        generated_text = generate_text(model, tgtokenizer, seed_text, next_words)

        return render_template('generate_result.html', seed_text=seed_text, generated_text=generated_text)

    return render_template('generate.html')

def generate_text(model, tgtokenizer, seed_text, next_words=20):
    
    # Set the maximum sequence length
    max_sequences_len = model.input_shape[1] + 1

    generated_text = seed_text

    for _ in range(next_words):
        token_list = tgtokenizer.texts_to_sequences([seed_text])[0]  # Use tgtokenizer
        token_list = pad_sequences([token_list], maxlen=max_sequences_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=1)
        output_word = ""

        for word, index in tgtokenizer.word_index.items():  # Use tgtokenizer
            if index == predicted:
                if word == '<OOV>':
                    output_word = ""
                else:
                    output_word = word
                break
            
        seed_text += " " + output_word
        generated_text += " " + output_word

    return generated_text






if __name__ == '__main__':
    app.run(debug=True)
