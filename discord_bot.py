import discord
from discord.ext import commands, tasks
from summarizer import summarize
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))
QNA_HOURS = int(os.getenv("QNA_HOURS", "2"))

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

def build_digest():
    summarized_news = summarize(refresh=True)

    yield f"**StackFeed Weekly AI Digest **"

    article_count = 0

    for category, articles in summarized_news.items():
        yield f"**{category.title()}**"
        for article in articles:
            article_count += 1

            yield (
                f"**{article['title']}**\n"
                f"{article['summary'].strip()}\n"
                f"Source: {article['source']}"
            )

    if article_count == 0:
        yield "No AI news found for this week."

async def send_digest():
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        channel = await bot.fetch_channel(CHANNEL_ID)

    await channel.send(f"**Preparing this week's AI Digest..**")

    messages = await asyncio.to_thread(lambda: list(build_digest()))

    for message in messages:
        await channel.send(message)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

    await send_digest()

    # print(f"Q&A window open for {QNA_HOURS} hours.")
    # await asyncio.sleep(QNA_HOURS * 60 * 60)
    #
    # print("Q&A window closed. Shutting down.")
    await bot.close()

bot.run(TOKEN)