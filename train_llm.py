import os
import numpy as np
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.initializers import Constant
import gensim.downloader as api
import pickle

# Set custom cache directory for Gensim
os.environ["GENSIM_DATA_DIR"] = "~/app/llm_env/cache/"
print(f"Gensim will use cache directory: {os.getenv('GENSIM_DATA_DIR')}")

# Function to download and load dataset
def download_dataset(url):
    """Download dataset from a URL and return it as a list of lines."""
    response = requests.get(url)
    response.raise_for_status()  # Ensure successful download
    return response.text.split("\n")

# Function to process datasets and prepare input-output pairs
def prepare_dataset(urls, sequence_length=10):
    """Download datasets, combine them, and prepare input-output sequences."""
    combined_text = ""
    
    # Download and combine datasets
    for url in urls:
        print(f"Downloading dataset from {url}...")
        dataset_lines = download_dataset(url)
        combined_text += " ".join(dataset_lines) + " "
    
    print(f"Combined dataset contains {len(combined_text.split())} words.")
    
    # Tokenize the combined text
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts([combined_text])
    sequences = tokenizer.texts_to_sequences([combined_text])[0]
    
    # Generate input-output pairs
    input_sequences = []
    output_sequences = []
    
    for i in range(len(sequences) - sequence_length):
        input_sequences.append(sequences[i:i + sequence_length])
        output_sequences.append(sequences[i + sequence_length])
    
    print(f"Number of input sequences generated: {len(input_sequences)}")
    if len(input_sequences) == 0:
        raise ValueError("No valid input sequences generated. Try reducing sequence_length or check your data.")
    
    return np.array(input_sequences), np.array(output_sequences), tokenizer

# Main script
if __name__ == "__main__":
    # Dataset URLs
    dataset_urls = [
        "https://www.gutenberg.org/cache/epub/345/pg345.txt",  # Dracula
        "https://www.gutenberg.org/cache/epub/56796/pg56796.txt"  # Additional dataset
    ]
    
    # Prepare datasets and sequences
    sequence_length = 10
    input_sequences, output_sequences, tokenizer = prepare_dataset(dataset_urls, sequence_length)
    
    vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
    print(f"Vocabulary size: {vocab_size}")
    
    # Load pre-trained GloVe word embeddings
    print("Loading GloVe embeddings...")
    word_vectors = api.load("glove-wiki-gigaword-100")  # 100D word embeddings
    
    # Create an embedding matrix
    embedding_dim = 100
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < vocab_size:
            embedding_vector = word_vectors[word] if word in word_vectors else None
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    # Define the LSTM model with pre-trained embeddings
    model = Sequential([
        Embedding(input_dim=vocab_size, 
                  output_dim=embedding_dim, 
                  embeddings_initializer=Constant(embedding_matrix), 
                  input_length=sequence_length, 
                  trainable=False),
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(vocab_size, activation='softmax')
    ])
    
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    # Train the model
    epochs = 30
    batch_size = 64
    print("Training the model...")
    model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)
    
    # Save the model and tokenizer
    model.save('trained_llm.h5')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("Model and tokenizer saved successfully.")
