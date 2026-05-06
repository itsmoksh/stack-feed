import discord
from discord.ext import commands, tasks
from summarizer import summarize
from rag.hybrid_rag import ingest_digest
from rag.rag_eval import EvalRag
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import os
import asyncio
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
DIGEST_CHANNEL_ID = int(os.getenv("DIGEST_CHANNEL_ID"))
METRICS_CHANNEL_ID = int(os.getenv("METRICS_CHANNEL_ID"))

QNA_HOURS = 2
TIME_ZONE = ZoneInfo("Asia/Kolkata")
qna_window_end = None
digest_started = False
qna_thread_ids = set()


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

def qna_window_is_open():
    return qna_window_end is not None and datetime.now(TIME_ZONE) < qna_window_end

# Get both the channels
async def get_digest_channel():
    channel = bot.get_channel(DIGEST_CHANNEL_ID)
    if channel is None:
        channel = await bot.fetch_channel(DIGEST_CHANNEL_ID)
    return channel

async def get_metrics_channel():
    channel = bot.get_channel(METRICS_CHANNEL_ID)
    if channel is None:
        channel = await bot.fetch_channel(METRICS_CHANNEL_ID)
    return channel

# Sends the weekly digest
async def send_digest():
    global qna_window_end
    channel = await get_digest_channel()

    await channel.send(f"Preparing this week's AI Digest..")
    summarized_news = summarize(refresh=True)
    await asyncio.to_thread(ingest_digest, refresh=True)
    article_count = 0
    for category, articles in summarized_news.items():

        # Send category header
        await channel.send(f"**{category.title()} updates**")

        for idx, article in enumerate(articles, start=1):
            article_count += 1
            embed = discord.Embed(
                title=f"{idx}. {article['title']}",
                description=article['summary'],
                color=0x5865F2
            )
            embed.set_footer(text=f"Source: {article['source']}")

            await channel.send(embed = embed)
    if article_count == 0:
        await channel.send("No AI news found for this week.")

    qna_window_end = datetime.now(TIME_ZONE) + timedelta(hours=QNA_HOURS)
    await channel.send(
        f"**\nQ&A is now open for the next {QNA_HOURS} hour(s). Ask question in this channel and I will answer you inside a thread.**"
    )

# Build the evaluation metrics after QnA
def build_metrics_embed():
    summary = EvalRag.summary()
    fallback_queries = summary['fallback_queries']
    low_score_queries = summary['low_score_queries']
    fallback_text = "None"
    if fallback_queries:
        fallback_text = "\n".join([f"- {query}" for query in fallback_queries[:10]])

    low_score_text = "None"
    if low_score_queries:
        low_score_text = "\n".join([
            f"- {item['metric_type']}: {item['score']:.2f} | {item['query']}"
            for item in low_score_queries[:10]
        ])

    embed = discord.Embed(
        title="RAG Evaluation Summary(Stack Feed)",
        description="RAG metrics for the latest bot run.",
        color=0x2ECC71,
        timestamp=datetime.now(TIME_ZONE)
    )
    embed.add_field(name="Total Questions", value=str(summary['total_queries']), inline=True)
    embed.add_field(name="Fallback Count", value=str(summary['fallback_score']), inline=True)
    embed.add_field(name="Avg Latency", value=f"{summary['average_total_latency']:.2f} sec", inline=True)
    embed.add_field(name="Avg Context Relevance", value=f"{summary['average_context_relevance']:.2f}", inline=True)
    embed.add_field(name="Avg Ground-ness", value=f"{summary['average_ground_ness']:.2f}", inline=True)
    embed.add_field(name="Avg Answer Relevance", value=f"{summary['average_answer_relevance']:.2f}", inline=True)
    embed.add_field(name="Fallback Queries", value=fallback_text[:1024], inline=False)
    embed.add_field(name="Low Score Queries (< 0.85)", value=low_score_text[:1024], inline=False)
    return embed

async def answer_in_thread(thread, question):
    eval_result = await asyncio.to_thread(EvalRag,question)
    await thread.send(eval_result.response)
    if eval_result.is_answerable:
        await thread.send(f"Sources: {eval_result.sources}")
    await asyncio.to_thread(eval_result.generate_report)

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
    metrics_channel = await get_metrics_channel()
    await digest_channel.send("This week's Q&A window is now closed. See you next Sunday.")
    await metrics_channel.send(embed=build_metrics_embed())
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

    if message.channel.id != DIGEST_CHANNEL_ID:
        return

    thread_name = f"Q/A for {message.author.display_name}"
    thread = await message.create_thread(name=thread_name)
    await answer_in_thread(thread, message.content)
    if thread.id not in qna_thread_ids:
        await thread.send(f"{message.author.mention} **If you have any further questions, Please ask in this thread only, this will keep the main chat clean.**")
    qna_thread_ids.add(thread.id)

if __name__ == "__main__":
    bot.run(TOKEN)