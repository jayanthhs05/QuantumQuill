import streamlit as st
import os
import datetime
from pathlib import Path
import traceback

# Import custom modules
from llm_utils import generate_story_chunk, get_available_models, unload_model
from story_manager import StoryManager
from ui_components import render_story_card, display_story_stats, render_help_section

# Set page configuration
st.set_page_config(
    page_title="QuantumQuill - AI Story Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    /* Dark mode theme */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    /* Consistent font family */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: #2D2D2D;
        color: #FFFFFF;
        border: 1px solid #404040;
        font-size: 1.1rem;
        line-height: 1.6;
        text-align: justify;
        white-space: pre-wrap;
        word-wrap: break-word;
        padding: 1.5rem;
        min-height: 600px;
    }
    
    /* Story content formatting */
    .story-content {
        text-align: justify;
        white-space: pre-wrap;
        word-wrap: break-word;
        line-height: 1.6;
        font-size: 1.1rem;
        padding: 1rem;
        background-color: #2D2D2D;
        border-radius: 4px;
        border: 1px solid #404040;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #4A4A4A;
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    
    .stButton button:hover {
        background-color: #5A5A5A;
        transform: translateY(-1px);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #252525;
    }
    
    /* Story cards */
    .story-card {
        background-color: #2D2D2D;
        border: 1px solid #404040;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .story-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .story-title {
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .story-meta, .story-stats {
        color: #B0B0B0;
        font-size: 0.9rem;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #2D2D2D;
        border-radius: 4px;
        padding: 1rem;
        border: 1px solid #404040;
    }
    
    /* Footer */
    .footer {
        color: #B0B0B0;
        text-align: center;
        padding: 1rem;
        font-size: 0.9rem;
        border-top: 1px solid #404040;
        margin-top: 2rem;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #1B4332;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 1rem;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #4A3C2A;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 1rem;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #1E3A8A;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize story manager
try:
    story_manager = StoryManager("stories")
except Exception as e:
    st.error(f"Failed to initialize story manager: {str(e)}")
    st.stop()

# Initialize session state variables if they don't exist
if 'current_story' not in st.session_state:
    try:
        st.session_state.current_story = story_manager.create_new_story()
    except Exception as e:
        st.error(f"Failed to create new story: {str(e)}")
        st.stop()

if 'editing_chunk' not in st.session_state:
    st.session_state.editing_chunk = False
if 'chunk_index' not in st.session_state:
    st.session_state.chunk_index = -1
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Function to start a new story
def start_new_story():
    try:
        new_story = story_manager.create_new_story()
        new_story["filename"] = None  # Ensure filename is None for new stories
        st.session_state.current_story = new_story
        st.session_state.editing_chunk = False
        st.session_state.chunk_index = -1
        st.rerun()
    except Exception as e:
        st.error(f"Failed to start new story: {str(e)}")
    
# Function to load a story
def load_story(story):
    try:
        st.session_state.current_story = story
        st.session_state.editing_chunk = False
        st.session_state.chunk_index = -1
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load story: {str(e)}")
    
# Function to delete a story
def delete_story(story):
    try:
        if story_manager.delete_story(story['filename']):
            st.success(f"Deleted '{story.get('title', 'Untitled Story')}'")
            st.rerun()
        else:
            st.error("Failed to delete story")
    except Exception as e:
        st.error(f"Error deleting story: {str(e)}")

# Function to save the current story
def save_current_story():
    try:
        if story_manager.save_story(st.session_state.current_story):
            st.success("Story saved successfully!")
        else:
            st.error("Failed to save story")
    except Exception as e:
        st.error(f"Error saving story: {str(e)}")

# Function to update story content with undo support
def update_story_content(new_content):
    try:
        # Save current content as last_generated before updating
        st.session_state.current_story["last_generated"] = st.session_state.current_story.get("content", "")
        st.session_state.current_story["content"] = new_content
        st.session_state.current_story["last_edited"] = datetime.datetime.now().isoformat()
    except Exception as e:
        st.error(f"Error updating story content: {str(e)}")

def clean_text_formatting(text):
    """
    Clean up text formatting by:
    1. Removing extra whitespace
    2. Fixing unwanted line breaks within sentences
    3. Ensuring proper paragraph spacing
    """
    if not text:
        return ""
    
    try:
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        # Clean each paragraph
        cleaned_paragraphs = []
        for para in paragraphs:
            # Remove extra whitespace and line breaks within sentences
            cleaned = ' '.join(para.split())
            cleaned_paragraphs.append(cleaned)
        
        # Join paragraphs with proper spacing
        return '\n\n'.join(cleaned_paragraphs)
    except Exception as e:
        st.error(f"Error cleaning text formatting: {str(e)}")
        return text  # Return original text if cleaning fails

# Function to handle model selection
def handle_model_selection():
    try:
        available_models = get_available_models()
        if not available_models:
            st.warning("No models found in the 'models' directory. Please add your models there.")
            return
            
        # Get the previously selected model
        previous_model = st.session_state.selected_model
        
        # Create model selection dropdown
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=available_models.index(previous_model) if previous_model in available_models else 0,
            key="model_selector"
        )
        
        # If model changed, unload the previous model
        if previous_model and previous_model != selected_model:
            unload_model(previous_model)
            
        st.session_state.selected_model = selected_model
        
    except Exception as e:
        st.error(f"Error handling model selection: {str(e)}")

# Main application UI
st.title("QuantumQuill - AI Story Generator")

# Sidebar for story management
with st.sidebar:
    st.title("QuantumQuill")
    
    # Model Selection
    st.subheader("Model Selection")
    handle_model_selection()
    
    st.markdown("---")
    
    # New Story Button
    if st.button("New Story", use_container_width=True):
        start_new_story()
    
    st.markdown("---")
    
    # Help section
    render_help_section()
    
    st.subheader("Your Stories")
    try:
        stories = story_manager.load_stories()
        
        if not stories:
            st.info("No saved stories yet. Create your first story!")
        
        for story in stories:
            render_story_card(story, load_story, delete_story)
    except Exception as e:
        st.error(f"Error loading stories: {str(e)}")

# Main content area
# Story title with larger font
st.markdown("<h2 style='margin-bottom: 0px;'>Story Title</h2>", unsafe_allow_html=True)
try:
    story_title = st.text_input("", value=st.session_state.current_story.get("title", ""), placeholder="Enter a title for your story")
    if story_title != st.session_state.current_story.get("title", ""):
        st.session_state.current_story["title"] = story_title
except Exception as e:
    st.error(f"Error updating story title: {str(e)}")

# Save button at the top
if st.button("💾 Save Story", use_container_width=True):
    save_current_story()

# Display story statistics
if st.session_state.current_story.get("content"):
    try:
        display_story_stats(st.session_state.current_story)
    except Exception as e:
        st.error(f"Error displaying story statistics: {str(e)}")

# Main content area
st.subheader("Your Story")
try:
    full_content = st.session_state.current_story.get("content", "")
    cleaned_content = clean_text_formatting(full_content)
    
    # Add editable text area for the story
    edited_content = st.text_area("", value=cleaned_content, height=600)
    
    # If content was edited, update the story
    if edited_content != cleaned_content:
        st.session_state.current_story["content"] = edited_content
        st.session_state.current_story["last_edited"] = datetime.datetime.now().isoformat()
    
    # Generate button
    if st.button("✨ Generate", use_container_width=True):
        if edited_content:
            with st.spinner("Generating story continuation..."):
                try:
                    # For new stories, use the entire content as prompt
                    # For existing stories, use the last paragraph as prompt
                    if not st.session_state.current_story.get("chunks"):
                        prompt = edited_content
                        previous_content = ""
                    else:
                        # Split into paragraphs and get the last one
                        paragraphs = edited_content.split('\n\n')
                        prompt = paragraphs[-1].strip()
                        previous_content = edited_content
                    
                    new_content = generate_story_chunk(prompt, previous_content, model_name=st.session_state.selected_model)
                    if new_content:
                        # Initialize chunks list if it doesn't exist
                        if "chunks" not in st.session_state.current_story:
                            st.session_state.current_story["chunks"] = []
                        
                        # Add the new chunk to the list
                        st.session_state.current_story["chunks"].append(new_content)
                        
                        # Update story content
                        if not st.session_state.current_story.get("chunks"):
                            # For first chunk, use the new content directly
                            st.session_state.current_story["content"] = new_content
                        else:
                            # For subsequent chunks, append to existing content
                            st.session_state.current_story["content"] = edited_content + "\n\n" + new_content
                        
                        # Update the model information
                        st.session_state.current_story["model"] = st.session_state.selected_model
                        
                        # Save the story without creating a new one
                        story_manager.save_story(st.session_state.current_story)
                        st.rerun()
                except Exception as e:
                    st.error(f"Error generating story continuation: {str(e)}")
        else:
            st.warning("Please start your story first!")
except Exception as e:
    st.error(f"Error handling story content: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    <p>QuantumQuill - AI Story Generator | Create, Edit, and Save Your Stories</p>
    <p style="font-size: 0.8em;">All stories are saved locally on your device</p>
</div>
""", unsafe_allow_html=True)