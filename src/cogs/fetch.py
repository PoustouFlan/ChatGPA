import discord
from discord import ui
from discord.ext import commands
from discord import app_commands
from bot_utils import guild_object

import logging
log = logging.getLogger("ChatGPA")

import pickle, weakref
from time import time

class PicklableMember:
    def __init__(self, member):
        self.name = member.name
        self.id = member.id
        self.nick = member.nick

class PicklableUser:
    def __init__(self, member):
        self.name = member.name
        self.id = member.id

class PicklableCategory:
    def __init__(self, category):
        self.name = category.name
        self.id = category.id

class PicklableChannel:
    def __init__(self, channel):
        self.name = channel.name
        self.category = PicklableCategory(channel.category)
        self.category_id = channel.category_id
        self.id = channel.id


class PicklableMessage:
    def __init__(self, message):
        self.content = message.content
        if type(message.author) == discord.Member:
            self.author = PicklableMember(message.author)
        else:
            self.author = PicklableUser(message.author)

        self.id = message.id
        if message.channel.type == discord.ChannelType.text:
            self.channel = PicklableChannel(message.channel)

class Fetch(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(
        name = "fetch",
        description = "vole la totalité des données du serveur pour les revendre à Google"
    )
    async def fetch(self, interaction):
        result = []
        checkpoint = time()
        for channel in await self.bot.guild.fetch_channels():
            if type(channel) == discord.channel.TextChannel:
                log.info(f"Fetching {channel}...")
                async for message in channel.history(limit = None):
                    if time() > checkpoint + 60:
                        log.debug(f"{len(result)} messages fetched so far.")
                        checkpoint = time()
                    result.append(message)
        
        try:
            result = list(map(PicklableMessage, result))
            with open("dump.pkl", 'wb') as f:
                pickle.dump(result, f)
            log.info("Success")
        except Exception as e:
            log.exception(str(e))



async def setup(bot):
    await bot.add_cog(Fetch(bot), guilds = [guild_object])
