import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, TimeDistributed, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ------------------------------------
# Configuration
# ------------------------------------

MODEL = "many_to_many.keras"
TOKENIZER = "tokenizer.pkl"
ENCODER = "label_encoder.pkl"

MAX_WORDS = 5000
MAX_LEN = 10

# ------------------------------------
# Train Model
# ------------------------------------

def train_model():

    df = pd.read_csv("spam.csv")

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df["sentence"])

    X = tokenizer.texts_to_sequences(df["sentence"])

    X = pad_sequences(
        X,
        maxlen=MAX_LEN,
        padding="post"
    )

    all_tags = []

    for tag in df["tags"]:
        all_tags.extend(tag.split())

    encoder = LabelEncoder()
    encoder.fit(all_tags)

    y = []

    for tag in df["tags"]:

        encoded = encoder.transform(tag.split())

        encoded = pad_sequences(
            [encoded],
            maxlen=MAX_LEN,
            padding="post"
        )[0]

        y.append(encoded)

    y = np.array(y)

    y = to_categorical(
        y,
        num_classes=len(encoder.classes_)
    )

    with open(TOKENIZER,"wb") as f:
        pickle.dump(tokenizer,f)

    with open(ENCODER,"wb") as f:
        pickle.dump(encoder,f)

    x_train,x_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=32,
            input_length=MAX_LEN
        )
    )

    model.add(
        SimpleRNN(
            64,
            return_sequences=True
        )
    )

    model.add(
        TimeDistributed(
            Dense(
                32,
                activation="relu"
            )
        )
    )

    model.add(
        TimeDistributed(
            Dense(
                len(encoder.classes_),
                activation="softmax"
            )
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=4,
        validation_data=(x_test,y_test)
    )

    model.save(MODEL)

# ------------------------------------
# Prediction
# ------------------------------------

def predict(sentence):

    model = load_model(MODEL)

    with open(TOKENIZER,"rb") as f:
        tokenizer = pickle.load(f)

    with open(ENCODER,"rb") as f:
        encoder = pickle.load(f)

    seq = tokenizer.texts_to_sequences([sentence])

    seq = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    prediction = model.predict(seq,verbose=0)

    prediction = np.argmax(
        prediction,
        axis=-1
    )[0]

    words = sentence.split()

    result=[]

    for i,word in enumerate(words):

        tag = encoder.inverse_transform(
            [prediction[i]]
        )[0]

        result.append((word,tag))

    return result

# ------------------------------------
# Train Model
# ------------------------------------

if not os.path.exists(MODEL):
    train_model()

# ------------------------------------
# Streamlit
# ------------------------------------

st.title("Many-to-Many RNN (POS Tagging)")

sentence = st.text_input("Enter Sentence")

if st.button("Predict"):

    if sentence.strip():

        output = predict(sentence)

        st.subheader("Predicted Tags")

        for word,tag in output:

            st.write(f"{word}  ➜  {tag}")

    else:

        st.warning("Please enter a sentence.")