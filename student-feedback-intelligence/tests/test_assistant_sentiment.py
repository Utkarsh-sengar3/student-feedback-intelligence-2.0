import unittest
from src.retrieval.rag_pipeline import RAGPipeline, parse_query_intent


class TestAssistantSentiment(unittest.TestCase):

    def test_parse_query_intent(self):
        # 1. Positive intent with count
        s, c, d, _ = parse_query_intent("give me 5 positive feedback")
        self.assertEqual(s, "Positive")
        self.assertEqual(c, 5)

        # 2. Negative intent with count
        s, c, d, _ = parse_query_intent("give me 5 negative feedback")
        self.assertEqual(s, "Negative")
        self.assertEqual(c, 5)

        # 3. Neutral intent with count
        s, c, d, _ = parse_query_intent("give me 5 neutral feedback")
        self.assertEqual(s, "Neutral")
        self.assertEqual(c, 5)

        # 4. Open-ended question without explicit sentiment filter
        s, c, d, _ = parse_query_intent("What do students say about teaching quality?")
        self.assertIsNone(s)
        self.assertIsNone(c)

    def test_rag_pipeline_sentiment_filtering(self):
        rag = RAGPipeline()

        # 1. Positive feedback query
        res_pos = rag.ask("give me 5 positive feedback")
        self.assertEqual(res_pos["retrieved_feedback_count"], 5)
        evidence_pos = res_pos["supporting_evidence"]
        self.assertEqual(len(evidence_pos), 5)
        self.assertTrue(all(e["sentiment"].lower() == "positive" for e in evidence_pos))
        self.assertFalse(any(e["sentiment"].lower() in ("negative", "neutral") for e in evidence_pos))

        # 2. Negative feedback query
        res_neg = rag.ask("give me 5 negative feedback")
        self.assertEqual(res_neg["retrieved_feedback_count"], 5)
        evidence_neg = res_neg["supporting_evidence"]
        self.assertEqual(len(evidence_neg), 5)
        self.assertTrue(all(e["sentiment"].lower() == "negative" for e in evidence_neg))
        self.assertFalse(any(e["sentiment"].lower() in ("positive", "neutral") for e in evidence_neg))

        # 3. Neutral feedback query
        res_neu = rag.ask("give me 5 neutral feedback")
        self.assertEqual(res_neu["retrieved_feedback_count"], 5)
        evidence_neu = res_neu["supporting_evidence"]
        self.assertEqual(len(evidence_neu), 5)
        self.assertTrue(all(e["sentiment"].lower() == "neutral" for e in evidence_neu))
        self.assertFalse(any(e["sentiment"].lower() in ("positive", "negative") for e in evidence_neu))

        # 4. Open-ended question (unrestricted semantic retrieval)
        res_open = rag.ask("What do students say about teaching quality?")
        self.assertGreater(res_open["retrieved_feedback_count"], 0)
        self.assertGreater(len(res_open["supporting_evidence"]), 0)


if __name__ == "__main__":
    unittest.main()
