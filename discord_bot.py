import discord
from discord.ext import commands
from processing.summarizer import summarize
from rag.vector_store import ingest_digest
from rag.workflow import build_graph
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from logging_setup import setup_logger
import os
import asyncio
load_dotenv()

bot_logger = setup_logger('bot_logger')

TOKEN = os.getenv("DISCORD_TOKEN")
DIGEST_CHANNEL_ID = int(os.getenv("DIGEST_CHANNEL_ID"))

CATEGORY_COLORS = {
    "model release": 0x5865F2,  # Blurple
    "product/tooling": 0xEB459E, # Gold
    "safety": 0xED4245,         # Red
    "research": 0x2ECC71        # Green
}

QNA_HOURS = 2
TIME_ZONE = ZoneInfo("Asia/Kolkata")
qna_window_end = None
digest_started = False
qna_thread_ids = set()


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
DISCORD_MESSAGE_LIMIT = 2000   # Discord's fixed cap on bot message content
rag_graph = build_graph(stream_generator=True)

def qna_window_is_open():
    return qna_window_end is not None and datetime.now(TIME_ZONE) < qna_window_end

# Get both the channels
async def get_digest_channel():
    channel = bot.get_channel(DIGEST_CHANNEL_ID)
    if channel is None:
        channel = await bot.fetch_channel(DIGEST_CHANNEL_ID)
    return channel

# Sends the weekly digest
async def send_digest():
    global qna_window_end
    digest_window_end = datetime.now(TIME_ZONE)
    digest_window_start = digest_window_end - timedelta(days=7)
    channel = await get_digest_channel()

    status_msg = await channel.send("Preparing this week's AI Digest..")
    story_count = 0
    category_count = 0
    async for category, summarized_articles in summarize(refresh=True):
        story_count += len(summarized_articles)
        category_count += 1
        category_title = f"{category.title()} updates"
        embed_color = CATEGORY_COLORS.get(category.lower().strip(), 0x7F8C8D)

        lines = []
        for idx, article in enumerate(summarized_articles, start=1):
            title = article.get('title', 'No Title')
            url = article.get('url')
            if url:
                lines.append(f"**{idx}.** [{title}]({url})")
            else:
                lines.append(f"**{idx}.** {title}")

        description = "\n\n".join(lines)

        embed = discord.Embed(
            title=category_title,
            description=description,
            color= CATEGORY_COLORS.get(category,0x7F8C8D)
        )
        embed.set_footer(text=f"{len(summarized_articles)} articles · full summaries in the thread")
        root_msg = await channel.send(embed=embed)
        category_thread =  await root_msg.create_thread(name = f"Summarized articles of {category}" )

        for idx, article in enumerate(summarized_articles, start=1):
            description = article['summary']
            if len(description) > 4096:
                description = description[:4093] + "..."
            summary_embed = discord.Embed(
                title=f"{idx}. {article['title']}",
                description=description,
                color=embed_color,
                url=article['source']
            )
            await category_thread.send(embed = summary_embed)

        await category_thread.edit(locked = True)

    date_range = (
        f"{digest_window_start:%d %b %Y} to {digest_window_end:%d %b %Y}"
    )
    await status_msg.edit(
        content=(
            f"AI Digest · 7-day window: {date_range} — "
            f"{story_count} stories across {category_count} categories"
        )
    )
    await asyncio.to_thread(ingest_digest, refresh=True)
    qna_window_end = datetime.now(TIME_ZONE) + timedelta(hours=QNA_HOURS)
    await channel.send(
        f"**Ask your questions here in the channel. Q&A is open for {QNA_HOURS} hours.**"
    )

def split_message(text, limit=DISCORD_MESSAGE_LIMIT):
    """
    Split text into chunks of at most `limit` characters, never cutting
    a word in half. Prefers paragraph breaks, then sentence ends, then
    line breaks, then plain word boundaries -- whichever keeps chunks
    most readable. Falls back to a hard cut only if a single "word"
    (e.g. a stray long URL) exceeds the limit on its own.
    """
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > limit:
        window = remaining[:limit]

        split_at = window.rfind('\n\n')          # paragraph break, first choice
        if split_at == -1:
            split_at = window.rfind('. ')         # sentence end, second choice
            if split_at != -1:
                split_at += 1
        if split_at == -1:
            split_at = window.rfind('\n')         # line break, third choice
        if split_at == -1:
            split_at = window.rfind(' ')          # any word boundary, last resort
        if split_at == -1:
            split_at = limit                      # unbreakable token -- unavoidable

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _stream_chunk_text(message_chunk):
    """Extract text from a streamed LangChain message chunk."""
    content = getattr(message_chunk, 'content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            block.get('text', '') if isinstance(block, dict) else str(block)
            for block in content
        )
    return ''


async def _update_answer_messages(thread, status_message, answer, messages, displayed):
    """Edit the live Discord messages to reflect the latest streamed answer."""
    parts = split_message(answer)
    for index, part in enumerate(parts):
        message = status_message if index == 0 else messages.get(index)
        if message is None:
            message = await thread.send(part)
            messages[index] = message
        elif displayed.get(index) != part:
            await message.edit(content=part)
        displayed[index] = part


async def answer_in_thread(thread, question):
    status_message = await thread.send('Routing your question and searching the articles…')
    answer = ''
    final_answer = ''
    messages = {}
    displayed = {0: 'Routing your question and searching the articles…'}
    last_update_text = displayed[0]
    last_flush = asyncio.get_running_loop().time()

    try:
        async for event in rag_graph.astream(
            {'question': question},
            stream_mode=['messages', 'updates'],
            version='v2',
        ):
            if event['type'] == 'updates':
                updates = event['data']
                if 'generator' in updates and isinstance(updates['generator'], dict):
                    final_answer = updates['generator'].get('answer') or final_answer

                progress_by_node = {
                    'router': 'Question routed; searching the articles…',
                    'category': 'Category identified; searching its articles…',
                    'decomposer': 'Breaking the comparison into focused searches…',
                    'retriever': 'Retrieving relevant article evidence…',
                    'evidence_gap': 'Checking evidence and searching for missing details…',
                    'category_scroll': 'Collecting the articles in this category…',
                    'combine_evidence': 'Evidence gathered; preparing your answer…',
                }
                if not answer:
                    for node_name, progress_text in progress_by_node.items():
                        if node_name in updates and progress_text != last_update_text:
                            await status_message.edit(content=progress_text)
                            last_update_text = progress_text

            elif event['type'] == 'messages':
                message_chunk, metadata = event['data']
                if metadata.get('langgraph_node') != 'generator':
                    continue

                answer += _stream_chunk_text(message_chunk)
                now = asyncio.get_running_loop().time()
                if answer and now - last_flush >= 0.75:
                    await _update_answer_messages(
                        thread,
                        status_message,
                        answer,
                        messages,
                        displayed,
                    )
                    last_flush = now

        final_answer = final_answer or answer
        if final_answer:
            await _update_answer_messages(
                thread,
                status_message,
                final_answer,
                messages,
                displayed,
            )
        elif not answer:
            await status_message.edit(content='I could not generate an answer for that question.')
    except Exception as e:
        bot_logger.exception('Failed to answer question %r: %s', question, e)
        error_text = 'Sorry, something went wrong while answering that. Please try asking again in a moment.'
        if answer:
            await thread.send(error_text)
        else:
            await status_message.edit(content=error_text)

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
    if qna_window_end is None:
        return
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
