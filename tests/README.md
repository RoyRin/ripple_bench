# Tests for Data to Concept Unlearning

This directory contains comprehensive tests for the WMDP Ripple Effect Pipeline and core ripple_bench functions.

## Test Structure

### `test_ripple_bench_split.py`
Tests for the split pipeline (dataset building and evaluation):
- **TestRippleBenchBuilder**: Unit tests for dataset building
  - `test_load_wmdp_questions`: Tests loading WMDP data from JSON files
  - `test_extract_topics_from_questions`: Tests topic extraction using LLMs
  - `test_caching_functionality`: Tests caching of intermediate results
  - `test_generate_questions_from_facts`: Tests question generation
  - `test_build_dataset_output_structure`: Tests complete dataset structure

- **TestRippleBenchEvaluator**: Unit tests for model evaluation
  - `test_load_dataset`: Tests loading ripple bench datasets
  - `test_evaluate_model`: Tests model evaluation on questions
  - `test_analyze_ripple_effects`: Tests ripple effect analysis
  - `test_create_visualizations`: Tests plot generation
  - `test_generate_report`: Tests evaluation report generation

- **TestIntegration**: Integration tests
  - `test_dataset_compatibility`: Verifies builder output works with evaluator

### `test_ripple_bench_core.py`
Tests for core functions used by the pipeline:
- **TestGenerateRippleQuestions**: Tests question generation utilities
  - `test_get_wmdp_question_answer`: Tests WMDP data parsing
  - `test_construct_single_dual_use_df_row`: Tests multiple choice formatting
  - `test_get_root_topic`: Tests topic extraction
  - `test_extract_bulleted_facts`: Tests fact extraction

- **TestUtils**: Tests utility functions
  - `test_save_and_read_dict`: Tests JSON I/O operations

- **TestOpenAIUtils**: Tests OpenAI API integration
  - `test_get_open_ai_huit_secret`: Tests secret loading
  - `test_make_openai_request_success`: Tests successful API calls
  - `test_make_openai_request_error`: Tests error handling

- **TestAnthropicUtils**: Tests Anthropic API integration
  - `test_get_anthropic_huit_secret`: Tests secret loading
  - `test_make_anthropic_request_success`: Tests API calls

- **TestDataValidation**: Tests edge cases and data validation
  - `test_empty_question_list`: Tests empty input handling
  - `test_malformed_wmdp_data`: Tests invalid data handling
  - `test_invalid_answer_index`: Tests index bounds checking
  - `test_unicode_handling`: Tests Unicode character support

## Running Tests

### Using the test runner script (recommended):
```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py -v

# Run only pipeline tests
python run_tests.py --pipeline

# Run only core function tests
python run_tests.py --core
```

### Using pytest directly:
```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_wmdp_ripple_pipeline.py -v

# Run specific test case
pytest tests/test_wmdp_ripple_pipeline.py::TestWMDPRippleEffectPipeline::test_load_wmdp_questions -v
```

### Using unittest directly:
```bash
# Run all tests
python -m unittest discover tests

# Run specific test module
python -m unittest tests.test_wmdp_ripple_pipeline

# Run specific test case
python -m unittest tests.test_wmdp_ripple_pipeline.TestWMDPRippleEffectPipeline.test_load_wmdp_questions
```

## Test Coverage

The tests use mocking extensively to avoid dependencies on:
- External APIs (OpenAI, Anthropic)
- Large language models
- Wikipedia API
- File system (except for temporary files)

This ensures tests run quickly and reliably without requiring:
- API keys
- Model weights
- Internet connection
- Specific file structures

## Writing New Tests

When adding new functionality:
1. Add unit tests for individual functions
2. Add integration tests for feature workflows
3. Use mocking for external dependencies
4. Test both success and failure cases
5. Include edge cases and validation tests

## Dependencies

The tests require:
- unittest (built-in)
- unittest.mock (built-in)
- tempfile (built-in)
- Optional: pytest for alternative test runner

No additional test dependencies are required beyond the main project dependencies.