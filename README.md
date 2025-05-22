# QuantumQuill

QuantumQuill is a local AI-powered creative writing assistant that helps you generate and manage stories using language models running on your own machine.

## Features

- Generate creative story continuations using local AI models
- Edit and refine AI-generated content
- Save and load your stories
- Track story statistics (word count, chunk count)
- Clean and consistent dark-themed UI
- All processing happens locally - no data sent to external servers

## Getting Started

### Prerequisites

- Python 3.9+
- GPU recommended for faster generation (but not required)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/QuantumQuill.git
   cd QuantumQuill
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Models setup:
   - Create a `models` directory if it doesn't exist
   - Download and add your language models to the `models` directory
   - Supported model formats:
     - Full model files (model.safetensors)
     - LoRA adapter models (adapter_model.safetensors)

## Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Project Structure

- `app.py` - Main Streamlit application
- `llm_utils.py` - Model loading and text generation utilities
- `story_manager.py` - Story saving, loading, and management
- `ui_components.py` - UI components for the Streamlit interface
- `test_models.py` - Script to test model compatibility
- `models/` - Place your models here, each in its own subdirectory
- `stories/` - Saved stories in JSON format

## Working with Models

The application supports various transformer-based language models:

- GPT-2 variants
- Pythia models
- GPT-Neo models
- And other HuggingFace compatible models

Models are loaded from the local `models` directory, with each model placed in its own subdirectory.

To test your models for compatibility:

```bash
python test_models.py
```

## Using QuantumQuill

1. **Select a model** from the dropdown in the sidebar
2. **Enter a title** for your story
3. **Type a seed sentence** in the text area to start your story
4. **Click "Generate"** to create the first chunk of your story
5. **Edit the text** as needed to refine the content
6. **Continue generating** to build your story
7. **Save your story** to revisit it later

## Development

The application is built with:

- [Streamlit](https://streamlit.io/) for the web interface
- [HuggingFace Transformers](https://huggingface.co/transformers/) for model handling
- [PyTorch](https://pytorch.org/) for machine learning operations

## Requirements

See `requirements.txt` for the full list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

_Note: QuantumQuill is designed to work with locally stored models to protect your privacy and give you full control over your creative writing process._
