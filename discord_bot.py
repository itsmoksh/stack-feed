import discord
from discord.ext import commands, tasks
from summarizer import summarize
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
CHANNEL_ID = int(os.getenv('CHANNEL_ID'))

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

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.command(name="send_ai_news")
async def send_weekly_digest(ctx):
    await ctx.channel.send("Preparing this week's AI Digest..")

    messages = await asyncio.to_thread(lambda: list(build_digest()))

    for message in messages:
        await ctx.channel.send(message)
        


async def send_ai_news(ctx):
    await send_weekly_digest(ctx.channel)

bot.run(TOKEN)