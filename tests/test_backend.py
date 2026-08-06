import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import numpy as np

# Set environment variables for testing
import os
os.environ["DEVICE"] = "cpu"

from app import app
from config import load_config
from inference import HateSpeechInference

class TestHateSpeechDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.config = load_config()

    def test_model_loading(self):
        # Test that inference engine can be instantiated and device is set
        engine = HateSpeechInference()
        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.tokenizer)
        self.assertIn(str(engine.device), ["cpu", "cuda"])

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_model_info_endpoint(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], self.config.model.model_name)
        self.assertEqual(data["labels"], ["Safe", "Offensive", "Hate Speech"])

    def test_predict_endpoint_success(self):
        response = self.client.post(
            "/predict",
            json={"text": "This is a clean and polite comment."}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)
        self.assertIn("processing_time_ms", data)
        self.assertIn(data["prediction"], ["Safe", "Offensive", "Hate Speech"])

    def test_predict_endpoint_validation(self):
        # Empty text
        response = self.client.post("/predict", json={"text": ""})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["detail"], "Validation error")
        self.assertIn("errors", data)

        # Missing key
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 422)

    def test_batch_predict_endpoint_success(self):
        response = self.client.post(
            "/batch-predict",
            json={"texts": ["I love coding.", "You are horrible!"]}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predictions", data)
        self.assertIn("total_processing_time_ms", data)
        self.assertEqual(len(data["predictions"]), 2)
        for pred in data["predictions"]:
            self.assertIn("prediction", pred)
            self.assertIn("confidence", pred)
            self.assertIn("probabilities", pred)

    def test_batch_predict_endpoint_validation(self):
        # Empty list
        response = self.client.post("/batch-predict", json={"texts": []})
        self.assertEqual(response.status_code, 422)

    @patch("requests.post")
    def test_voice_client_api_flow(self, mock_post):
        # Mock API success response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": "Safe",
            "confidence": 99.1,
            "probabilities": {"Safe": 99.1, "Offensive": 0.5, "Hate Speech": 0.4},
            "processing_time_ms": 15.0
        }
        mock_post.return_value = mock_response

        # Instantiate voice interface client
        from voice import VoiceInterface
        client = VoiceInterface(api_url="http://127.0.0.1:8000")
        
        # Mock speak method so it doesn't try to audio output
        client.speak = MagicMock()
        
        # Run prediction
        client.run_prediction("Hello, world!", use_local_fallback=False)
        
        # Verify post called
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8000/predict",
            json={"text": "Hello, world!"},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        # Verify speech output was triggered
        client.speak.assert_called_once_with("Predicted category is Safe with a confidence of 99.10 percent.")

    @patch("requests.post")
    def test_voice_client_fallback_flow(self, mock_post):
        # Simulate connection error to API
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        from voice import VoiceInterface
        client = VoiceInterface(api_url="http://127.0.0.1:8000")
        client.speak = MagicMock()
        client._run_local_prediction = MagicMock()

        # Run prediction with local fallback enabled
        client.run_prediction("Test local fallback text.", use_local_fallback=True)

        # Verify fallback called
        client._run_local_prediction.assert_called_once_with("Test local fallback text.")

if __name__ == "__main__":
    unittest.main()
