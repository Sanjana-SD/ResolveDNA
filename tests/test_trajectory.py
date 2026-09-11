"""Unit tests for trajectory parsing and reconstruction."""
import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.parser import detect_brand_actions, RES_REGEX, COMPLAINT_REGEX


class TestBrandActionDetection:
    def test_detect_dm_request(self):
        text = "Please send us a DM with your order details"
        actions = detect_brand_actions(text)
        assert "request_dm" in actions

    def test_detect_provide_link(self):
        text = "Try the steps here: https://amazon.com/help/video"
        actions = detect_brand_actions(text)
        assert "provide_link" in actions

    def test_detect_troubleshoot(self):
        text = "Please try to restart your device and clear the app cache"
        actions = detect_brand_actions(text)
        assert "troubleshoot" in actions

    def test_detect_clarify_info(self):
        text = "Could you share your order number so we can look into this?"
        actions = detect_brand_actions(text)
        assert "clarify_info" in actions

    def test_detect_apologize(self):
        text = "We're really sorry to hear about this experience"
        actions = detect_brand_actions(text)
        assert "apologize" in actions

    def test_detect_general_response(self):
        text = "Hello! How can we help you today?"
        actions = detect_brand_actions(text)
        assert "general_response" in actions

    def test_multiple_actions(self):
        text = "Sorry about that! Please restart your device and try again"
        actions = detect_brand_actions(text)
        assert "apologize" in actions
        assert "troubleshoot" in actions


class TestResolutionSignals:
    def test_resolution_positive(self):
        assert RES_REGEX.search("Thanks, that fixed it!")
        assert RES_REGEX.search("Working now, appreciate the help!")
        assert RES_REGEX.search("Got it, thank you!")

    def test_resolution_negative(self):
        assert not RES_REGEX.search("This is terrible service")
        assert not RES_REGEX.search("Why is my order late?")

    def test_complaint_positive(self):
        assert COMPLAINT_REGEX.search("Still not working after trying that")
        assert COMPLAINT_REGEX.search("I already called and no response")
        assert COMPLAINT_REGEX.search("This is useless, worst experience ever")

    def test_complaint_negative(self):
        assert not COMPLAINT_REGEX.search("Thanks for the quick help!")
