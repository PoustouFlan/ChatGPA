import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

import json
import pickle

from random import sample, shuffle
from collections import Counter, defaultdict

## DATASET PREPARATION
print("Loading data")

data = pickle.load(open('data/dump.pkl', 'rb'))

id_author = defaultdict(lambda: '') # maps id to username
counter = Counter([message.author.name for message in data])
authors = counter.most_common(19) # we will use only the 19 most common posters
message_count = counter[authors[-1][0]]

author_messages = defaultdict(lambda: []) # maps a user to ≈1,000 of his messages
for message in data:
    author_messages[
        message.author.name
    ].append(message)
    id_author[message.author.id] = message.author.name

filtered_data = []
for author, _ in authors:
    if author in ('Ballsdex', 'UwU', 'Jacks',
                  'Shyn3ss_Bot', 'Rythm'): # We get rid of bots
        continue
    filtered_data.extend(
        list(map(
            lambda message: (message.content, author),
            sample(author_messages[author], message_count)
        ))
    )

shuffle(filtered_data)

messages = [message[0] for message in filtered_data]
authors = [message[1] for message in filtered_data]

## TEXT PREPROCESSING

def preprocess_text(text):
    text = text.lower()
    # Replace mention by actual username
    mention_pattern = r"<@!?([a-zA-Z0-9_]+)>"
    text = re.sub(mention_pattern, lambda match: id_author[int(match.group(1))], text)
    # Replace emoji by emoji name
    emoji_pattern = r"<:([a-zA-Z0-9_]+):[a-zA-Z0-9_]+>"
    text = re.sub(emoji_pattern, lambda match: match.group(1), text)
    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text

messages = [preprocess_text(message) for message in messages]

## LABEL CONVERSION
author_set = list(set(authors))
author_to_id = {author: i for i, author in enumerate(author_set)}
labels = [author_to_id[author] for author in authors]

with open('data/author_set.pkl', 'wb') as f:
    pickle.dump(author_set, f)
    pickle.dump(dict(id_author), f)

## TOKENIZATION AND WORD EMBEDDINGS
tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

sequences = tokenizer.texts_to_sequences(messages)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

print("FastText loading")
fasttext_path = 'data/cc.fr.300.vec'
word_embeddings = KeyedVectors.load_word2vec_format(fasttext_path, binary=False)


embedding_dim = 300

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if word in word_embeddings:
        embedding_matrix[i] = word_embeddings[word]

tokenizer_json = tokenizer.to_json()
json.dump(tokenizer_json, open('data/tokenizer_config.json', 'w'))

## SPLITTING THE DATASET INTO TRAINING/VALIDATION SETS
validation_split = 0.2
num_validation_samples = int(validation_split * len(padded_sequences))

indices = np.arange(len(padded_sequences))
np.random.shuffle(indices)
padded_sequences = padded_sequences[indices]
labels = np.array(labels)[indices]

X_train = padded_sequences[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
X_val = padded_sequences[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

## BUILDING THE MODEL
hidden_units = 64

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
    tf.keras.layers.LSTM(hidden_units, return_sequences=True),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(hidden_units),
    tf.keras.layers.Dense(len(author_set), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

## TRAINING
epochs = 50
batch_size = 32

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])

model.save("data/chatGPA_model.h5")

print("Complete.")
