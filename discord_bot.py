import discord
from discord.ext import commands, tasks
from summarizer import summarize
from rag import ingest_digest, get_relevant_qa
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import os
import asyncio
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

QNA_HOURS = 2
TIME_ZONE = ZoneInfo("Asia/Kolkata")
qna_window_end = None
digest_started = False
qna_thread_ids = set()


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

def build_digest():
    summarized_news = summarize(refresh=False)
    ingest_digest(refresh=True)

    article_count = 0

    for category, articles in summarized_news.items():
        yield f"** Weekly Updates about {category.title()}**"

        for idx, article in enumerate(articles, start=1):
            article_count += 1

            yield (
                f"\n**{idx}. {article['title']}**"
                f"{article['summary'].strip()}\n"
                f"*Source:* {article['source']}\n\n"
            )

    if article_count == 0:
        yield "No AI news found for this week."

def qna_window_is_open():
    return qna_window_end is not None and datetime.now(TIME_ZONE) < qna_window_end

async def get_digest_channel():
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        channel = bot.fetch_channel(CHANNEL_ID)
    return channel

async def send_digest():
    global qna_window_end
    channel = await get_digest_channel()
    await channel.send(f"**Preparing this week's AI Digest..**")

    messages = await asyncio.to_thread(lambda: list(build_digest()))

    for message in messages:
        await channel.send(message)

    qna_window_end = datetime.now(TIME_ZONE) + timedelta(hours=QNA_HOURS)
    await channel.send(
        f"**\nQ&A is now open for the next {QNA_HOURS} hour(s). Ask question in this channel and I will answer you inside a thread.**"
    )

async def answer_in_thread(thread, question):
    answer = await asyncio.to_thread(get_relevant_qa, question)
    await thread.send(answer)

async def close_all_qna_threads():
    for thread_id in qna_thread_ids:
        thread = bot.get_channel(thread_id)

        if thread is None:
            try:
                thread = await bot.fetch_channel(thread_id)
            except:
                continue

        if isinstance(thread, discord.Thread):
            await thread.send("**\nHope your queries are resolved, we are closing this thread!**")
            await thread.edit(archived=True, locked=True)

async def close_bot_after_qna_window():
    while qna_window_end is None:
        await asyncio.sleep(1)

    remaining_seconds = (qna_window_end - datetime.now(TIME_ZONE)).total_seconds()
    if remaining_seconds > 0:
        await asyncio.sleep(remaining_seconds)

    digest_channel = await get_digest_channel()
    await digest_channel.send("This week's Q&A window is now closed. See you next Sunday.")
    await close_all_qna_threads()
    await bot.close()

@bot.event
async def on_ready():
    global digest_started

    print(f"Logged in as {bot.user}")
    if digest_started:
        return

    digest_started = True
    await send_digest()
    bot.loop.create_task(close_bot_after_qna_window())

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if not qna_window_is_open():
        return

    if isinstance(message.channel, discord.Thread):
        if message.channel.id in qna_thread_ids:
            await answer_in_thread(message.channel, message.content)
        return

    if message.channel.id != CHANNEL_ID:
        return

    thread_name = f"Q/A for {message.author.display_name}"
    thread = await message.create_thread(name=thread_name)
    await answer_in_thread(thread, message.content)
    if thread.id not in qna_thread_ids:
        await thread.send(f"{message.author.mention} **If you have any further questions, Please ask in this thread only, this will keep the main chat clean.**")
    qna_thread_ids.add(thread.id)

bot.run(TOKEN)