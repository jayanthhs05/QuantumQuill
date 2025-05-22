import streamlit as st
import datetime

def render_story_card(story, on_load, on_delete):
    """
    Render a story card in the sidebar with load and delete options.
    
    Args:
        story (dict): Story data
        on_load (function): Callback when story is loaded
        on_delete (function): Callback when story is deleted
    """
    title = story.get("title", "Untitled Story")
    created_at = story.get("created_at", "")
    word_count = len(story.get("content", "").split())
    model = story.get("model", "Unknown Model")
    
    # Calculate chunk count from chunks list
    chunk_count = len(story.get("chunks", []))
    
    # Format date
    if created_at:
        try:
            created_date = datetime.datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M")
        except:
            created_date = ""
    else:
        created_date = ""
    
    # Create card with CSS
    card_html = f"""
    <style>
    .story-card {{
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }}
    .story-title {{
        font-weight: bold;
        margin-bottom: 5px;
    }}
    .story-meta {{
        color: #666;
        font-size: 0.8em;
        margin-bottom: 5px;
    }}
    .story-stats {{
        display: flex;
        justify-content: space-between;
        font-size: 0.8em;
        color: #666;
    }}
    .story-model {{
        color: #666;
        font-size: 0.8em;
        font-style: italic;
        margin-top: 5px;
    }}
    </style>
    
    <div class="story-card">
        <div class="story-title">{title}</div>
        <div class="story-meta">Created: {created_date}</div>
        <div class="story-stats">
            <span>{word_count} words</span>
            <span>{chunk_count} chunks</span>
        </div>
        <div class="story-model">Model: {model}</div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Load", key=f"load_{story['filename']}"):
            on_load(story)
    with col2:
        if st.button("🗑️", key=f"delete_{story['filename']}"):
            on_delete(story)

def display_story_stats(story):
    """
    Display statistics about the current story.
    
    Args:
        story (dict): Story data
    """
    content = story.get("content", "")
    chunks = story.get("chunks", [])
    
    # Calculate statistics
    word_count = len(content.split())
    char_count = len(content)
    chunk_count = len(chunks)
    avg_chunk_size = round(word_count / max(chunk_count, 1))
    
    # Display in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Words", word_count)
    with col2:
        st.metric("Characters", char_count)
    with col3:
        st.metric("Chunks", chunk_count)
    with col4:
        st.metric("Avg. Words/Chunk", avg_chunk_size)

def render_help_section():
    """
    Render a help section with instructions.
    """
    with st.expander("How to Use QuantumQuill"):
        st.markdown("""
        ### Getting Started
        1. **Enter a seed sentence** in the text area on the right
        2. **Click 'Generate Next Chunk'** to create the first part of your story
        3. **Edit the generated text** if needed
        4. **Click 'Apply Changes'** to save your edits
        5. **Continue adding chunks** to build your story
        6. **Save your story** when you're happy with it
        
        ### Tips
        - Be specific in your prompts for better results
        - Edit generated content to maintain consistency
        - Save your work frequently
        - Use the story history to revisit previous creations
        """)