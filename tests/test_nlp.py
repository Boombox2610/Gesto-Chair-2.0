import unittest
from unittest.mock import MagicMock

from src.nlp_engine.nlp_config import config
from src.nlp_engine.reasoner import SceneReasoner
from src.nlp_engine.vlm import VLMEngine


class TestNLPEngine(unittest.TestCase):
    def setUp(self):
        self.mock_ingester = MagicMock()
        self.mock_payload = {
            "schema_version": "1.0",
            "frame_id": 100,
            "fps": 28.5,
            "scene_understanding": [
                {
                    "object_id": 1,
                    "class": "Chair",
                    "position": "Center",
                    "depth_score": 0.65,
                    "priority": "standard",
                },
                {
                    "object_id": 2,
                    "class": "Wet floor",
                    "position": "Center-Left",
                    "depth_score": 0.85,
                    "priority": "critical",
                },
            ],
            "recognized_faces": [
                {"name": "Alice", "object_id": 10}
            ],
            "environmental_text": [
                {"text": "EXIT", "confidence": 0.95}
            ],
        }
        self.mock_ingester.history = [self.mock_payload]
        self.mock_vlm = MagicMock()
        self.reasoner = SceneReasoner(self.mock_ingester, vlm=self.mock_vlm)

    def test_intent_classification(self):
        self.assertEqual(self.reasoner._classify_intent("what's around me"), "SCENE_SUMMARY")
        self.assertEqual(self.reasoner._classify_intent("are there any hazards"), "HAZARD_CHECK")
        self.assertEqual(self.reasoner._classify_intent("is anyone nearby"), "PEOPLE_CHECK")
        self.assertEqual(self.reasoner._classify_intent("read that sign"), "OCR_READ")
        self.assertEqual(self.reasoner._classify_intent("what's the fps"), "STATUS")
        self.assertEqual(self.reasoner._classify_intent("switch to hindi"), "LANG_SWITCH")
        self.assertEqual(self.reasoner._classify_intent("what color is the shirt"), "VLM_QUERY")
        self.assertEqual(self.reasoner._classify_intent("something unusual here"), "UNKNOWN")

    def test_hazard_check_handler(self):
        response = self.reasoner.answer("any hazards?", language="en")
        self.assertIn("Wet floor", response)
        self.assertIn("Yes", response)

    def test_people_check_handler(self):
        response = self.reasoner.answer("is anyone nearby?", language="en")
        self.assertIn("Alice", response)

    def test_ocr_read_handler(self):
        response = self.reasoner.answer("read the sign", language="en")
        self.assertIn("EXIT", response)

    def test_status_handler(self):
        response = self.reasoner.answer("what is the fps", language="en")
        self.assertIn("28.5", response)

    def test_vlm_escalation(self):
        self.mock_vlm.answer.return_value = "The person is wearing a blue jacket."
        config.vlm_enabled = True
        
        response = self.reasoner.answer("what color is the jacket?", language="en")
        self.mock_vlm.answer.assert_called_once()
        self.assertEqual(response, "The person is wearing a blue jacket.")

    def test_vlm_context_builder(self):
        vlm = VLMEngine()
        context = vlm._build_context_string(self.mock_payload)
        self.assertIn("Chair", context)
        self.assertIn("Wet floor", context)
        self.assertIn("Alice", context)
        self.assertIn("EXIT", context)


if __name__ == "__main__":
    unittest.main()
