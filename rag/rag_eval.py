from dotenv import load_dotenv
from groq import Groq
from rag.hybrid_rag import retrieve_chunks,generate_response
import time
from collections import Counter
import threading
from logging_setup import setup_logger
from groq_retry import run_groq_with_retry
from pathlib import Path
load_dotenv()

log_path = Path(__file__).parent.parent/'stack_feed.log'
groq_client = Groq()
eval_logger = setup_logger('eval_logger',log_path)

class EvalRag:
    metrics = {'total_latency':[],'context_relevance':[],'ground_ness':[],'answer_relevance':[],'low_cr_queries':[],'low_gr_queries':[],'low_ar_queries':[]}
    metrics_lock = threading.Lock()

    def __init__(self,query):
        self.query = query
        self.response, self.context,self.sources = self.evaluate_latency_and_generate()

    def evaluate_latency_and_generate(self):
        start_time = time.time()
        chunk_ids,sources, context = retrieve_chunks(self.query)
        response = generate_response(self.query,context)
        end_time = time.time()
        total_latency = end_time - start_time
        eval_logger.info(f"Latency for {self.query}: {total_latency}")
        self._add_metric('total_latency', float(total_latency))
        return response.choices[0].message.content, context, sources

    @classmethod
    def _add_metric(cls, metric_name, value):
        with cls.metrics_lock:
            cls.metrics[metric_name].append(value)

    @classmethod
    def _track_problematic_events(cls,query,context_rel,groundedness,answer_rel):
        with cls.metrics_lock:
            if context_rel == 'LOW':
                cls.metrics['low_cr_queries'].append(query)
                eval_logger.warning(f'"Low Context Relevance"  for {query}')

            if groundedness == 'UNSUPPORTED':
                cls.metrics['low_gr_queries'].append(query)
                eval_logger.warning(f'"UNSUPPORTED Groundedness" for {query}')

            if answer_rel == 'IRRELEVANT':
                cls.metrics['low_ar_queries'].append(query)
                eval_logger.warning(f'"IRRELEVANT Answer Relevance"  for {query}')


    @staticmethod
    def _label_from_response(response):
        content = response.choices[0].message.content.strip().upper()
        ordered_labels = [
            "PARTIALLY SUPPORTED",
            "PARTIALLY RELEVANT",
            "UNSUPPORTED",
            "SUPPORTED",
            "IRRELEVANT",
            "RELEVANT",
            "MEDIUM",
            "HIGH",
            "LOW"
        ]

        for label in ordered_labels:

            if content == label:
                return label

        eval_logger.debug(
            f"Could not parse evaluation label from: {content}"
        )
        return "UNKNOWN"


    def context_relevance(self):
        con_rel_prompt = f'''
        You are an evaluator for a RAG system.
        Classify whether the context can answer the query.
        Labels:
        HIGH:
        - Context fully contains the information needed.
        MEDIUM:
        - Context partially answers the query.
        LOW:
        - Context does not contain enough information.

        Context:
        {self.context}

        Query:
        {self.query}

        Output only one label:
        HIGH, MEDIUM, or LOW
        '''
        def make_call():
            return groq_client.chat.completions.create(
                model='openai/gpt-oss-120b',
                messages=[{"role": "user", "content": con_rel_prompt}],
                temperature=0)
        context_relevance_score = run_groq_with_retry(make_call)
        label = self._label_from_response(context_relevance_score)
        eval_logger.info(f'Context relevance: "{label}" for {self.query}')
        self._add_metric('context_relevance', label)
        return label

    def ground_ness(self):
        ground_ness_prompt = f'''<|im_start|>system
        You are an auditing algorithm. Verify if a response is explicitly supported by the context.
        Output only labels: 
        SUPPORTED: If the response is fully supported by the context.
        PARTIALLY_SUPPORTED: If the response is somewhere supported by the context.
        UNSUPPORTED: If the response is not at all supported by the context.<|im_end|>system
        <|im_start|>user
        Context:
        {self.context}

        Answer:
        {self.response}
        
        Output ONLY one label:
        SUPPORTED
        PARTIALLY SUPPORTED
        UNSUPPORTED
        <|im_end|>user'''

        def make_call():
            return groq_client.chat.completions.create(
                model='openai/gpt-oss-120b',
                messages=[{"role": "user", "content": ground_ness_prompt}],
                temperature=0)
        grounded_score = run_groq_with_retry(make_call)

        label = self._label_from_response(grounded_score)
        eval_logger.info(f'Groundedness: "{label}" for {self.query}')
        self._add_metric('ground_ness', label)
        return label

    def answer_relevance(self):
        ans_rel_prompt = f'''
        You are an evaluator for a RAG system.
        Determine whether the answer addresses the user's query appropriately.
        
        Labels:
        RELEVANT:
        - Answer directly addresses the query.
        PARTIALLY RELEVANT:
        - Answer partially addresses the query.
        IRRELEVANT:
        - Answer does not address the query properly.
        Important:
        If the context does not contain enough information and the answer correctly refuses to answer, classify it as RELEVANT.

        Query:
        {self.query}

        Answer:
        {self.response}

        Output only one label:
        RELEVANT, PARTIALLY RELEVANT, or IRRELEVANT
        '''

        def make_call():
            return groq_client.chat.completions.create(
                model='openai/gpt-oss-120b',
                messages=[{"role": "user", "content": ans_rel_prompt}],
                temperature=0)
        ans_relevance_score = run_groq_with_retry(make_call)

        label = self._label_from_response(ans_relevance_score)
        eval_logger.info(f'Answer relevance: "{label}" for {self.query}')
        self._add_metric('answer_relevance', label)
        return label



    def generate_report(self):
        context_rel = self.context_relevance()
        groundedness = self.ground_ness()
        ans_rel = self.answer_relevance()

        self._track_problematic_events(
            self.query,context_rel,groundedness,ans_rel
        )

    @classmethod
    def summary(cls):
        with cls.metrics_lock:
            total_queries = len(cls.metrics['total_latency'])
            avg_latency = (sum(cls.metrics['total_latency']) / total_queries if total_queries > 0 else 0)
            return {
                'total_queries': total_queries,
                'average_total_latency': avg_latency,
                'context_relevance_distribution': dict(Counter(cls.metrics['context_relevance'])),
                'ground_ness_distribution': dict(Counter(cls.metrics['ground_ness'])),
                'answer_relevance_distribution': dict(Counter(cls.metrics['answer_relevance'])),
                'low_cr_queries': list(cls.metrics['low_cr_queries']),
                'low_gr_queries': list(cls.metrics['low_gr_queries']),
                'low_ar_queries': list(cls.metrics['low_ar_queries'])
            }



