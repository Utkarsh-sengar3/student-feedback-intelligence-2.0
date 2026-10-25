import unittest
from api.services.analytics_service import AnalyticsService


class TestAnalyticsService(unittest.TestCase):

    def setUp(self):
        self.service = AnalyticsService.get_instance()

    def test_overview_unfiltered(self):
        res = self.service.get_overview()
        kpis = res["kpis"]
        self.assertEqual(kpis["total_records"], 6500)
        self.assertEqual(kpis["positive_count"], 3405)
        self.assertEqual(kpis["negative_count"], 1602)
        self.assertEqual(kpis["neutral_count"], 1493)
        self.assertAlmostEqual(kpis["positive_pct"] + kpis["negative_pct"] + kpis["neutral_pct"], 100.0, delta=0.5)
        self.assertEqual(len(res["sentiment_distribution"]), 3)
        self.assertGreater(len(res["domain_distribution"]), 0)
        self.assertGreater(len(res["category_volume"]), 0)
        self.assertFalse(res["temporal_notice"]["has_temporal_data"])

    def test_overview_filtered_by_domain(self):
        res = self.service.get_overview(domain="teacher review")
        kpis = res["kpis"]
        self.assertEqual(kpis["total_records"], 3000)
        self.assertEqual(kpis["distinct_domains"], 1)

    def test_overview_filtered_by_sentiment(self):
        res = self.service.get_overview(sentiment="negative")
        kpis = res["kpis"]
        self.assertEqual(kpis["total_records"], 1602)
        self.assertEqual(kpis["negative_count"], 1602)
        self.assertEqual(kpis["positive_count"], 0)
        self.assertEqual(kpis["neutral_count"], 0)

    def test_overview_filtered_by_category(self):
        res = self.service.get_overview(category="Staff - Teaching")
        kpis = res["kpis"]
        self.assertGreater(kpis["total_records"], 0)

    def test_records_pagination(self):
        res = self.service.get_records(page=1, page_size=10)
        self.assertEqual(res["total"], 6500)
        self.assertEqual(len(res["records"]), 10)
        self.assertEqual(res["page"], 1)
        self.assertEqual(res["total_pages"], 650)
        first = res["records"][0]
        self.assertIn("review_id", first)
        self.assertIn("categories", first)
        self.assertIn("text", first)
        self.assertIn("overall_sentiment", first)

    def test_export_csv(self):
        csv_data = self.service.export_csv(domain="university review")
        lines = csv_data.strip().splitlines()
        # Header + 500 rows = 501 lines
        self.assertEqual(len(lines), 501)
        self.assertIn("review_id,domain,overall_sentiment", lines[0])


if __name__ == "__main__":
    unittest.main()
