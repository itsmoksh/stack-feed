from dotenv import load_dotenv
from groq import Groq
from rag.hybrid_rag import retrieve_chunks,generate_response
import time
import re
import threading
from logging_setup import setup_logger
from pathlib import Path
load_dotenv()

log_path = Path(__file__).parent.parent/'stack_feed.log'
groq_client = Groq()
eval_loger = setup_logger('eval_loger',log_path)

class EvalRag:
    LOW_SCORE_THRESHOLD = 0.85
    metrics = {'total_latency':[],'context_relevance_score':[],'ground_ness_score':[],'answer_relevance_score':[],'fallback_score':0,'fallback_queries':[],'low_score_queries':[]}
    metrics_lock = threading.Lock()

    def __init__(self,query):
        self.query = query
        self.response, self.context,self.sources = self.evaluate_latency_and_generate()
        self.is_answerable = self.answerable()

    def evaluate_latency_and_generate(self):
        start_time = time.time()
        chunk_ids,sources, context = retrieve_chunks(self.query)
        response = generate_response(self.query,context)
        end_time = time.time()
        total_latency = end_time - start_time
        self._add_metric('total_latency', float(total_latency))
        return response.choices[0].message.content, context, sources

    @classmethod
    def _add_metric(cls, metric_name, value):
        with cls.metrics_lock:
            cls.metrics[metric_name].append(value)

    @classmethod
    def _add_fallback(cls, query):
        with cls.metrics_lock:
            cls.metrics["fallback_score"] += 1
            cls.metrics["fallback_queries"].append(query)

    @classmethod
    def _add_low_score_query(cls, metric_type, query, score):
        if score >= cls.LOW_SCORE_THRESHOLD:
            return

        with cls.metrics_lock:
            cls.metrics["low_score_queries"].append({
                "metric_type": metric_type,
                "query": query,
                "score": score,
            })
            eval_loger.debug(f'Got lower {metric_type} score: {score} for {query}')

    @staticmethod
    def _score_from_response(response):
        content = response.choices[0].message.content.strip()
        match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", content)
        if match:
            return float(match.group(0))
        eval_loger.warning(f"Could not parse evaluation score from: {content}")
        return 0.0

    @staticmethod
    def _average(values):
        if not values:
            return 0.0
        return sum(values) / len(values)

    def answerable(self):
        no_context_phrases = ["no context",
        "not found",
        "no relevant",
        "no information",
        "unable to find"
        ]

        is_answerable = not any(phrase in self.response.lower() for phrase in no_context_phrases)

        return is_answerable

    def context_relevance(self):
        con_rel_prompt = f'''
            You are an expert evaluator. Rate if this context can answer the query.
            Output between the range of 0 to 1(probabilities): 
            -Higher probability if a context is able to answer the given query.
            -Lower probability if a context is not able to answer the given query
            Context: {self.context}
            Query: {self.query}
            Output only probabilities, no reasoning
            '''
        context_relevance_score = groq_client.chat.completions.create(
            model='openai/gpt-oss-120b',
            messages=[{"role": "user", "content": con_rel_prompt}],
            temperature=0)
        score = self._score_from_response(context_relevance_score)
        self._add_metric('context_relevance_score', score)
        self._add_low_score_query('context_relevance', self.query, score)

    def ground_ness(self):
        ground_ness_prompt = f'''
            You are an auditing algorithm. Verify if a value is explained by the given context.
            Output between the range of 0 to 1(probabilities): 
            - Higher probability if a value is from the give context only.
            - Lower probability if a value is not from the give context.
            Context:
            {self.context}
            Value to verify:
            {self.response}
            Answer only score, no justification'''

        grounded_score = groq_client.chat.completions.create(
            model='openai/gpt-oss-120b',
            messages=[{"role": "user", "content": ground_ness_prompt}],
            temperature=0)

        score = self._score_from_response(grounded_score)
        self._add_metric('ground_ness_score', score)
        self._add_low_score_query('ground_ness', self.query, score)

    def answer_relevance(self):
        ans_rel_prompt = f'''
            You are an expert evaluator for a RAG-based question answering system.
            Your task is to judge ANSWER RELEVANCE,  whether the answer directly addresses the user's question, stays on topic, and provides information at the appropriate scope.
            Output between the range of 0 to 1(probabilities):
            - Higher probability if answers the question with the right scope and no unnecessary off-topic content.
            - Lower probability if do not answer the question with the right scope and is off-topic.
            Query: {self.query}
            Answer: {self.response}
            Output only probabilities, no reasoning'''

        ans_relevance_score = groq_client.chat.completions.create(
            model='openai/gpt-oss-120b',
            messages=[{"role": "user", "content": ans_rel_prompt}],
            temperature=0)

        score = self._score_from_response(ans_relevance_score)
        self._add_metric('answer_relevance_score', score)
        self._add_low_score_query('answer_relevance', self.query, score)


    def generate_report(self):
        if  self.is_answerable:
            self.context_relevance()
            self.ground_ness()
            self.answer_relevance()
        else:
            self._add_fallback(self.query)
            eval_loger.debug(f'No context found in the articles {self.query}')

    @classmethod
    def summary(cls):
        with cls.metrics_lock:
            total_queries = len(cls.metrics['total_latency'])
            return {
                'total_queries': total_queries,
                'average_total_latency': cls._average(cls.metrics['total_latency']),
                'average_context_relevance': cls._average(cls.metrics['context_relevance_score']),
                'average_ground_ness': cls._average(cls.metrics['ground_ness_score']),
                'average_answer_relevance': cls._average(cls.metrics['answer_relevance_score']),
                'fallback_score': cls.metrics['fallback_score'],
                'fallback_queries': list(cls.metrics['fallback_queries']),
                'low_score_queries': list(cls.metrics['low_score_queries']),
            }



