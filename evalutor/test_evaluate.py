import unittest
from evalutor.evaluate import TextEvaluator
import nltk



nltk.download('punkt')

class TestTextEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = TextEvaluator()

    def test_evaluate_all(self):
        question = "What are the effects of global warming?"
        response = "Global warming leads to rising sea levels and extreme weather events."
        reference = "The effects of global warming include rising sea levels, more extreme weather events, and loss of biodiversity."
        metrics = self.evaluator.evaluate_all(question, response, reference)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("BLEU", metrics)
        self.assertIn("ROUGE-1", metrics)
        self.assertIn("BERT P", metrics)
        self.assertIn("BERT R", metrics)
        self.assertIn("BERT F1", metrics)
        self.assertIn("Flesch Reading Ease", metrics)
        self.assertIn("Flesch-Kincaid Grade", metrics)
        self.assertIn("Diversity", metrics)
        self.assertIn("Racial Bias", metrics)
        self.assertIn("MAUVE", metrics)
        self.assertIn("Flesch Reading Ease", metrics)
        self.assertIn("Flesch-Kincaid Grade", metrics)

if __name__ == "__main__":
    unittest.main()