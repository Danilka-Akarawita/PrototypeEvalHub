import torch
import mauve
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline,AutoTokenizer, AutoModel
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sklearn.metrics.pairwise import cosine_similarity
from mauve import compute_mauve
import nltk


class TextEvaluator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModel.from_pretrained("gpt2")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.bias_pipeline = pipeline("zero-shot-classification", model="Hate-speech-CNERG/dehatebert-mono-english")

    def load_gpt2_model(self):
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        return model, tokenizer
    
    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    
    def evaluate_bleu_rouge(self, candidates, references):
        bleu_score = corpus_bleu(candidates, [references]).score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
        rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        return bleu_score, rouge1
    
    def evaluate_bert_score(self, reference, candidate):
        P, R, F1 = score(candidate, reference, lang='en',model_type='bert-base-multilingual-cased', verbose=True)
        return P.mean().item(), R.mean().item(), F1.mean().item()
    
    def evaluate_racial_bias(self, text):
        results = self.bias_pipeline([text], candidate_labels=["hate speech", "not hate speech"])
        bias_score = results[0]['scores'][results[0]['labels'].index('hate speech')]
        return bias_score
    
    def evaluate_readability(self, text):
        flesch_score = flesch_reading_ease(text)
        flesch_grade = flesch_kincaid_grade(text)
        return flesch_score, flesch_grade 
    
    def evaluate_mauve(self, reference, candidate):
        ref_embedding = self.get_embeddings([reference])
        cand_embedding = self.get_embeddings([candidate])
        return compute_mauve(ref_embedding, cand_embedding, verbose=False).mauve
    
    def evaluate_diversity(self, texts):
        all_tokens = [tok for text in texts for tok in text.split()]
        unique_bigrams = set(ngrams(all_tokens, 2))
        diversity_score = len(unique_bigrams) / len(all_tokens) if all_tokens else 0
        return diversity_score

    def evaluate_all(self, question, response, reference):
        candidates = [response]
        references = [reference]
        bleu, rouge1 = self.evaluate_bleu_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
        flesch_ease, flesch_grade = self.evaluate_readability(response)
        diversity = self.evaluate_diversity(candidates)
        racial_bias = self.evaluate_racial_bias(response)
        mauve_score = self.evaluate_mauve(reference, response)
        flesch_ease, flesch_grade = self.evaluate_readability(response)
        return {
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "BERT P": bert_p,
            "BERT R": bert_r,
            "BERT F1": bert_f1,
            "Flesch Reading Ease": flesch_ease,
            "Flesch-Kincaid Grade": flesch_grade,
            "Diversity": diversity,
            "Racial Bias": racial_bias,
            "MAUVE": mauve_score,
            "Flesch Reading Ease": flesch_ease,
            "Flesch-Kincaid Grade": flesch_grade,
        }