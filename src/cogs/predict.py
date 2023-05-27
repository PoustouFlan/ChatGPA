import discord
from discord import ui
from discord.ext import commands
from discord import app_commands
from bot_utils import guild_object

import logging
log = logging.getLogger("ChatGPA")

import re

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json

model = tf.keras.models.load_model("data/chatGPA_model.h5")
with open('data/author_set.pkl', 'rb') as f:
    author_set = pickle.load(f)
    id_author = pickle.load(f)

data = json.load(open('data/tokenizer_config.json', 'r'))
tokenizer = tokenizer_from_json(data)

def preprocess_text(text):
    text = text.lower()
    mention_pattern = r"<@!?([a-zA-Z0-9_]+)>"
    text = re.sub(mention_pattern, lambda match: id_author[int(match.group(1))], text)
    emoji_pattern = r"<:([a-zA-Z0-9_]+):[a-zA-Z0-9_]+>"

    text = re.sub(emoji_pattern, lambda match: match.group(1), text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text

class Predict(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(
        name = "predict",
        description = "Prédit qui aurait envoyé un tel message"
    )
    async def predict(self, interaction, message:str):
        print("Predicting")
        try:
            preprocessed_message = preprocess_text(message)

            sequence = tokenizer.texts_to_sequences([preprocessed_message])
            max_sequence_length = model.input_shape[1]
            padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

            prediction = model.predict(padded_sequence)
            predicted_author_id = tf.argmax(prediction, axis=1).numpy()[0]
            predicted_author = author_set[predicted_author_id]
        except Exception as e:
            print(e)

        print(predicted_author)

        await interaction.response.send_message(
            f"Je pense que le message « {message} » a été envoyé par {predicted_author}."
        )

async def setup(bot):
    await bot.add_cog(Predict(bot), guilds = [guild_object])
