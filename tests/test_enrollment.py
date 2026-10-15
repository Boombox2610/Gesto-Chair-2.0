import unittest
from unittest.mock import MagicMock, patch
import time

from src.nlp_engine.enrollment import EnrollmentController
from src.nlp_engine.nlp_config import config


class TestEnrollmentController(unittest.TestCase):
    def setUp(self):
        self.controller = EnrollmentController(db_path="data/test_faces.pkl")
        self.controller.faces_dir = "data/test_sample_faces"
        config.enrollment_enabled = True
        config.enrollment_depth_thresh = 0.5
        config.enrollment_cooldown = 60

    def test_initial_state(self):
        self.assertEqual(self.controller.state, EnrollmentController.STATE_IDLE)
        self.assertFalse(self.controller.is_active())

    @patch("src.nlp_engine.enrollment.tts_manager.speak")
    def test_unknown_person_trigger(self, mock_speak):
        payload = {
            "scene_understanding": [
                {
                    "object_id": 105,
                    "class": "Person",
                    "position": "Center-Left",
                    "depth_score": 0.75,
                    "bbox": [100, 100, 300, 400],
                }
            ],
            "recognized_faces": []
        }

        self.controller.process_payload(payload)
        self.assertEqual(self.controller.state, EnrollmentController.STATE_AWAIT_CONFIRMATION)
        self.assertTrue(self.controller.is_active())
        self.assertEqual(self.controller.current_candidate["object_id"], 105)
        mock_speak.assert_called_once()
        self.assertIn("Unknown person", mock_speak.call_args[0][0])

    @patch("src.nlp_engine.enrollment.tts_manager.speak")
    def test_user_cancellation(self, mock_speak):
        self.controller.state = EnrollmentController.STATE_AWAIT_CONFIRMATION
        self.controller.current_candidate = {"object_id": 105, "position": "Center-Left"}

        handled = self.controller.handle_voice_input("No thanks", language="en")
        self.assertTrue(handled)
        self.assertEqual(self.controller.state, EnrollmentController.STATE_IDLE)
        self.assertFalse(self.controller.is_active())
        self.assertIn("cancelled", mock_speak.call_args[0][0].lower())

    @patch("src.nlp_engine.enrollment.tts_manager.speak")
    def test_user_confirmation_and_naming(self, mock_speak):
        self.controller.state = EnrollmentController.STATE_AWAIT_CONFIRMATION
        self.controller.current_candidate = {"object_id": 105, "position": "Center-Left"}

        # User says yes
        handled = self.controller.handle_voice_input("Yes", language="en")
        self.assertTrue(handled)
        self.assertEqual(self.controller.state, EnrollmentController.STATE_AWAIT_NAME)
        self.assertIn("name", mock_speak.call_args[0][0].lower())

        # Mock the execute enrollment method to avoid webcam/cv2 operations
        with patch.object(self.controller, "_execute_enrollment", return_value=(True, "Rahul has been added.")):
            handled_name = self.controller.handle_voice_input("His name is Rahul", language="en")
            self.assertTrue(handled_name)
            self.assertEqual(self.controller.state, EnrollmentController.STATE_IDLE)
            self.assertFalse(self.controller.is_active())

    @patch("src.nlp_engine.enrollment.tts_manager.speak")
    def test_question_passthrough_cancels_enrollment(self, mock_speak):
        self.controller.state = EnrollmentController.STATE_AWAIT_CONFIRMATION
        self.controller.current_candidate = {"object_id": 105, "position": "Center-Left"}

        # User asks a scene question instead of yes/no
        handled = self.controller.handle_voice_input("What is happening in this scene?", language="en")
        self.assertFalse(handled) # Returns False so run_nlp routes to SceneReasoner
        self.assertEqual(self.controller.state, EnrollmentController.STATE_IDLE)
        self.assertFalse(self.controller.is_active())

    @patch("src.nlp_engine.enrollment.tts_manager.speak")
    def test_cooldown_prevents_immediate_retrigger(self, mock_speak):
        payload = {
            "scene_understanding": [
                {"object_id": 105, "class": "Person", "depth_score": 0.8, "position": "Ahead"}
            ],
            "recognized_faces": []
        }

        # First trigger
        self.controller.process_payload(payload)
        self.assertEqual(self.controller.state, EnrollmentController.STATE_AWAIT_CONFIRMATION)

        # Cancel
        self.controller._cancel()

        # Second trigger for same object_id immediately after should NOT fire
        mock_speak.reset_mock()
        self.controller.process_payload(payload)
        self.assertEqual(self.controller.state, EnrollmentController.STATE_IDLE)
        mock_speak.assert_not_called()

    @patch("src.nlp_engine.enrollment.tts_manager.speak")
    def test_invalid_name_rejected(self, mock_speak):
        self.controller.state = EnrollmentController.STATE_AWAIT_NAME
        self.controller.current_candidate = {"object_id": 105, "position": "Center-Left"}

        # User says long gibberish / monologue
        handled = self.controller.handle_voice_input("i want to get the phone what are your names phakkar kota i am", language="en")
        self.assertTrue(handled)
        self.assertEqual(self.controller.state, EnrollmentController.STATE_AWAIT_NAME)
        self.assertIn("first and last name", mock_speak.call_args[0][0].lower())

    @patch("src.nlp_engine.enrollment.tts_manager.speak")
    def test_global_cooldown_prevents_retrigger_different_id(self, mock_speak):
        payload1 = {
            "scene_understanding": [
                {"object_id": 105, "class": "Person", "depth_score": 0.8, "position": "Ahead"}
            ],
            "recognized_faces": []
        }
        payload2 = {
            "scene_understanding": [
                {"object_id": 200, "class": "Person", "depth_score": 0.8, "position": "Ahead"}
            ],
            "recognized_faces": []
        }

        # First trigger on ID 105
        self.controller.process_payload(payload1)
        self.assertEqual(self.controller.state, EnrollmentController.STATE_AWAIT_CONFIRMATION)
        self.controller._cancel()

        # Immediate trigger on NEW ID 200 within global cooldown (45s) must be blocked
        mock_speak.reset_mock()
        self.controller.process_payload(payload2)
        self.assertEqual(self.controller.state, EnrollmentController.STATE_IDLE)
        mock_speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
