import discord
import asyncio
from bot_utils import *
from discord.ext import commands

import logging
log = logging.getLogger("ChatGPA")
log.setLevel(logging.DEBUG)
stream = logging.StreamHandler()
log.addHandler(stream)

class Bot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.synced = True
        self.guild = None


intents = discord.Intents.default()
intents.message_content = True
bot = Bot(
    command_prefix = "/",
    help_command = None,
    intents = intents
)

initial_extensions = [
    "cogs.fetch",
    "cogs.predict",
]

@bot.event
async def on_ready():
    log.info(f"Connecté en tant que {bot.user}")
    if not bot.synced:
        log.info("Syncing...")
        fmt = await bot.tree.sync(guild = guild_object)
        s = "" if len(fmt) < 2 else "s"
        log.info("Sync complete")
        log.info(f"{len(fmt)} commande{s} synchronisée{s}.")
        bot.synced = True

    bot.guild = bot.get_guild(GUILD_ID)



async def load():
    for extension in initial_extensions:
        try:
            await bot.load_extension(extension)
            log.info(f"{extension} chargée avec succès")
        except Exception as e:
            log.error(f"Échec du chargement de {extension}")
            log.error(e)


async def main():
    await load()
    await bot.start(TOKEN)

asyncio.run(main())
