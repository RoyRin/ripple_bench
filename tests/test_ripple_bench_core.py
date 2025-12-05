#!/usr/bin/env python3
"""
Tests for core ripple_bench functions used by the pipeline

Run with: python -m pytest tests/test_ripple_bench_core.py -v
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ripple_bench.generate_ripple_questions import (
    get_wmdp_question_answer, 
    construct_single_dual_use_df_row,
    get_root_topic,
    extract_bulleted_facts
)
from ripple_bench.utils import save_dict, read_dict
from ripple_bench.openai_utils import get_open_ai_huit_secret, make_openai_request
from ripple_bench.anthropic_utils import get_anthropic_huit_secret, make_anthropic_request, anthropic_function


class TestGenerateRippleQuestions(unittest.TestCase):
    """Test functions from generate_ripple_questions module"""
    
    def test_get_wmdp_question_answer(self):
        """Test extracting question-answer pairs from WMDP data"""
        wmdp_data = [
            {
                "question": "Which toxin causes paralysis?",
                "choices": ["Botulinum", "Tetanus", "Ricin", "Sarin"],
                "answer": 0
            }
        ]
        
        questions, answers = get_wmdp_question_answer(wmdp_data)
        
        self.assertEqual(len(questions), 1)
        self.assertEqual(len(answers), 1)
        self.assertEqual(questions[0], "Which toxin causes paralysis?")
        self.assertEqual(answers[0], "Botulinum")
    
    def test_construct_single_dual_use_df_row(self):
        """Test construction of shuffled multiple choice questions"""
        question = "Which toxin causes paralysis?"
        answer = "Botulinum"
        choices = ["Botulinum", "Tetanus", "Ricin", "Sarin"]
        
        # Test multiple times to ensure randomization works
        results = []
        for _ in range(5):
            row = construct_single_dual_use_df_row(question, answer, choices)
            results.append(row)
        
        # Check structure
        for row in results:
            self.assertIn('question', row)
            self.assertIn('answer', row)
            self.assertIn('answer_index', row)
            self.assertIn('choices', row)
            self.assertEqual(len(row['choices']), 4)
            
            # Verify answer is in choices
            self.assertIn(answer, row['choices'])
            
            # Verify answer_index is correct
            answer_idx = row['answer_index']
            self.assertEqual(row['choices'][answer_idx], answer)
            self.assertIn(answer_idx, [0, 1, 2, 3])
    
    @patch('ripple_bench.generate_ripple_questions.huit_OAI_function')
    def test_get_root_topic(self, mock_oai):
        """Test topic extraction from questions"""
        question = "Which bacteria produces botulinum toxin?"
        
        # Mock OpenAI response
        mock_oai.return_value = "Botulinum toxin"
        
        topic = get_root_topic(question)
        
        self.assertEqual(topic, "Botulinum toxin")
        
        # Verify prompt was constructed correctly
        mock_oai.assert_called_once()
        call_args = mock_oai.call_args[0][0]
        self.assertIn(question, call_args)
        self.assertIn("wikipedia", call_args.lower())
    
    @patch('ripple_bench.generate_ripple_questions.generate_text')
    def test_extract_bulleted_facts(self, mock_generate):
        """Test fact extraction from text"""
        passage = "Botulinum toxin is produced by Clostridium botulinum. It causes paralysis."
        
        # Mock model response
        mock_generate.return_value = "• Botulinum toxin is produced by Clostridium botulinum\n• It causes paralysis"
        
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        facts = extract_bulleted_facts(passage, mock_model, mock_tokenizer, max_new_tokens=100)
        
        self.assertIn("Clostridium botulinum", facts)
        self.assertIn("paralysis", facts)
        self.assertIn("•", facts)


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.temp_path = Path(self.temp_file.name)
    
    def tearDown(self):
        self.temp_path.unlink()
    
    def test_save_and_read_dict(self):
        """Test saving and reading dictionaries"""
        test_data = {
            "key1": "value1",
            "key2": [1, 2, 3],
            "key3": {"nested": "dict"}
        }
        
        # Test save
        save_dict(test_data, str(self.temp_path))
        
        # Test read
        loaded_data = read_dict(str(self.temp_path))
        
        self.assertEqual(loaded_data, test_data)
        self.assertEqual(loaded_data["key1"], "value1")
        self.assertEqual(loaded_data["key2"], [1, 2, 3])
        self.assertEqual(loaded_data["key3"]["nested"], "dict")


class TestOpenAIUtils(unittest.TestCase):
    """Test OpenAI utility functions"""
    
    @patch('builtins.open', create=True)
    @patch('ripple_bench.openai_utils.SECRET_DIR')
    def test_get_open_ai_huit_secret(self, mock_secret_dir, mock_open):
        """Test reading OpenAI secret"""
        mock_secret_dir.__truediv__.return_value = Path("fake/path")
        mock_open.return_value.__enter__.return_value.read.return_value = "test_api_key\n"
        
        key = get_open_ai_huit_secret()
        
        self.assertEqual(key, "test_api_key")
    
    @patch('requests.request')
    def test_make_openai_request_success(self, mock_request):
        """Test successful OpenAI API request"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            "choices": [{
                "message": {
                    "content": "Test response"
                }
            }]
        })
        mock_request.return_value = mock_response
        
        result = make_openai_request("Test prompt", "test_key")
        
        self.assertEqual(result, "Test response")
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        self.assertEqual(call_args[0][0], "POST")
        self.assertIn("chat/completions", call_args[0][1])
        
        # Check headers
        headers = call_args[1]['headers']
        self.assertEqual(headers['api-key'], "test_key")
    
    @patch('requests.request')
    def test_make_openai_request_error(self, mock_request):
        """Test OpenAI API request error handling"""
        # Mock error response
        mock_request.side_effect = Exception("API Error")
        
        result = make_openai_request("Test prompt", "test_key")
        
        self.assertIsNone(result)


class TestAnthropicUtils(unittest.TestCase):
    """Test Anthropic utility functions"""
    
    @patch('builtins.open', create=True)
    @patch('ripple_bench.anthropic_utils.SECRET_DIR')
    def test_get_anthropic_huit_secret(self, mock_secret_dir, mock_open):
        """Test reading Anthropic secret"""
        mock_secret_dir.__truediv__.return_value = Path("fake/path")
        mock_open.return_value.__enter__.return_value.read.return_value = "test_anthropic_key\n"
        
        key = get_anthropic_huit_secret()
        
        self.assertEqual(key, "test_anthropic_key")
    
    @patch('requests.request')
    def test_make_anthropic_request_success(self, mock_request):
        """Test successful Anthropic API request"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            "content": [{
                "text": "Claude response"
            }]
        })
        mock_request.return_value = mock_response
        
        result = make_anthropic_request("Test prompt", "test_key", max_tokens=100)
        
        self.assertEqual(result, "Claude response")
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        self.assertEqual(call_args[0][0], "POST")
        self.assertIn("messages", call_args[0][1])
        
        # Check payload
        payload = json.loads(call_args[1]['data'])
        self.assertEqual(payload['max_tokens'], 100)
        self.assertIn('claude', payload['model'])


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases"""
    
    def test_empty_question_list(self):
        """Test handling of empty question lists"""
        questions, answers = get_wmdp_question_answer([])
        self.assertEqual(questions, [])
        self.assertEqual(answers, [])
    
    def test_malformed_wmdp_data(self):
        """Test handling of malformed WMDP data"""
        # Missing 'answer' field
        malformed_data = [{
            "question": "Test?",
            "choices": ["A", "B", "C", "D"]
        }]
        
        with self.assertRaises(KeyError):
            get_wmdp_question_answer(malformed_data)
    
    def test_invalid_answer_index(self):
        """Test handling of invalid answer indices"""
        wmdp_data = [{
            "question": "Test?",
            "choices": ["A", "B"],
            "answer": 5  # Invalid index
        }]
        
        with self.assertRaises(IndexError):
            get_wmdp_question_answer(wmdp_data)
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters in questions"""
        question = "What is the β-lactam structure?"
        answer = "β-lactam ring"
        choices = ["β-lactam ring", "α-helix", "γ-globulin", "δ-toxin"]
        
        row = construct_single_dual_use_df_row(question, answer, choices)
        
        self.assertIn("β-lactam", row['question'])
        self.assertIn(answer, row['choices'])


if __name__ == '__main__':
    unittest.main()