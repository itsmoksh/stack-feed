import discord
from discord.ext import commands
from processing.summarizer import summarize
from rag.hybrid_rag import ingest_digest
from rag.rag_eval import EvalRag
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from pathlib import Path
from logging_setup import setup_logger
import os
import asyncio
load_dotenv()

log_path = Path(__file__).parent / 'stack_feed.log'
bot_logger = setup_logger('bot_logger', log_path)

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
    summarized_news = await asyncio.to_thread(summarize,refresh=True)
    sorted_news = dict(sorted(summarized_news.items()))
    await asyncio.to_thread(ingest_digest, refresh=True)
    article_count = 0
    for category, articles in sorted_news.items():

        if category == 'General':
            await channel.send(f"**General/Overall updates this week**")
        elif category == 'newsletter':
            await channel.send(f"**Published Newsletters this week**")
        else:
            await channel.send(f"**{category.title()} updates**")

        for idx, article in enumerate(articles, start=1):
            article_count += 1
            try:
                description = article['summary']
                if len(description) > 4096:
                    description = description[:4093] + "..."
                embed = discord.Embed(
                    title=f"{idx}. {article['title']}",
                    description=description,
                    color=0x5865F2
                )
                embed.set_footer(text=f"Source: {article['source']}")

                await channel.send(embed = embed)
            except Exception as e:
                bot_logger.error(f"Failed to post article '{article.get('title')}': {e}")
    if article_count == 0:
        await channel.send("No AI news found for this week.")

    qna_window_end = datetime.now(TIME_ZONE) + timedelta(hours=QNA_HOURS)
    await channel.send(
        f"**\nQ&A is now open for the next {QNA_HOURS} hour(s). Ask question in this channel and I will answer you inside a thread.**"
    )

# Build the evaluation metrics after QnA
def build_metrics_embed():
    summary = EvalRag.summary()
    total_queries = max(summary['total_queries'], 1)
    context_relevance_distribution = summary['context_relevance_distribution']
    ground_ness_distribution = summary['ground_ness_distribution']
    answer_relevance_distribution = summary['answer_relevance_distribution']
    low_cr_queries = summary['low_cr_queries']
    low_gr_queries = summary['low_gr_queries']
    low_ar_queries = summary['low_ar_queries']

    embed = discord.Embed(
        title="RAG Evaluation Summary (Stack Feed)",
        description="RAG metrics for the latest bot run.",
        color=0x2ECC71,
        timestamp=datetime.now(TIME_ZONE)
    )
    embed.add_field(name="Total Questions", value=str(summary['total_queries']), inline=True)
    embed.add_field(name="Avg Latency", value=f"{summary['average_total_latency']:.2f} sec", inline=True)

    if context_relevance_distribution:
        embed.add_field(name= 'Context Relevance Distribution', value = "\n".join(
            [f"{category.title()}: {round((query_count*100)/total_queries)}%" for category,query_count in context_relevance_distribution.items()]),
                        inline = False)

    if ground_ness_distribution:
        embed.add_field(name = 'Groundedness Distribution', value = "\n".join(
            [f"{category.title()}: {round((query_count * 100) /total_queries)}%" for category, query_count
             in ground_ness_distribution.items()]),
                        inline = False)

    if answer_relevance_distribution:
        embed.add_field(name= 'Answer Relevance Distribution', value = "\n".join(
            [f"{category.title()}: {round((query_count * 100) /total_queries)}%" for category, query_count
             in answer_relevance_distribution.items()]),
                        inline = False)

    if low_cr_queries:
        embed.add_field(name = 'Low Context Relevance Queries',
                        value= "\n".join(low_cr_queries)[:1024],
                        inline = False)

    if low_gr_queries:
        embed.add_field(name = 'Unsupported Groundedness Queries',
                       value= "\n".join(low_gr_queries)[:1024],
                       inline = False)

    if low_ar_queries:
        embed.add_field(name = 'Irrelevant Answer Queries',
                        value = "\n".join(low_ar_queries)[:1024],
                        inline = False)

    return embed

async def answer_in_thread(thread, question):
    try:
        eval_result = await asyncio.to_thread(EvalRag,question)
        await thread.send(eval_result.response)
        await thread.send(f"Sources: {eval_result.sources}")
        await asyncio.to_thread(eval_result.generate_report)
    except Exception as e:
        bot_logger.error(f"Failed to answer question '{question}': {e}")
        await thread.send("Sorry, something went wrong while answering that. Please try asking again in a moment.")

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
    try:
        await send_digest()
    except Exception as e:
        bot_logger.error(f"Digest generation failed: {e}")
        try:
            channel = await get_digest_channel()
            await channel.send("Something went wrong while preparing this week's digest. Please check the logs.")
        except Exception:
            pass
        await bot.close()
        return

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