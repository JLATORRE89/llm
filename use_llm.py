import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import gensim.downloader as api

# Set custom cache directory for Gensim
os.environ["GENSIM_DATA_DIR"] = "~/app/llm_env/cache/"
print(f"Gensim will use cache directory: {os.getenv('GENSIM_DATA_DIR')}")

# Load the trained model
model = load_model('trained_llm.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Function to generate text
def generate_text(seed_text, model, tokenizer, sequence_length, num_words_to_generate=30):
    generated_text = seed_text

    for _ in range(num_words_to_generate):
        token_list = tokenizer.texts_to_sequences([generated_text])
        token_list = pad_sequences(token_list, maxlen=sequence_length, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]

        output_word = tokenizer.index_word.get(predicted_word_index, "")
        if output_word:
            generated_text += " " + output_word
        else:
            break

    return generated_text

# Test text generation
seed_text = "It is a truth universally acknowledged"
sequence_length = 10  # Match the sequence length used during training
generated_text = generate_text(seed_text, model, tokenizer, sequence_length, num_words_to_generate=50)
print("Generated Text:")
print(generated_text)
