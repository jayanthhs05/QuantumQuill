import json
import datetime
from pathlib import Path
import streamlit as st

class StoryManager:
    """
    Class to handle story management operations including saving, loading, and manipulating stories.
    """
    def __init__(self, stories_dir="stories"):
        self.stories_dir = Path(stories_dir)
        self.stories_dir.mkdir(exist_ok=True)
    
    def save_story(self, story):
        """
        Save a story to a JSON file.
        
        Args:
            story (dict): Story data including title, content, and chunks
            
        Returns:
            bool: Success status
        """
        try:
            # Use existing filename if available, otherwise create new one
            if not story.get("filename"):
                # Create a filename based on the title and timestamp for new stories
                if not story.get("title"):
                    story["title"] = "Untitled Story"
                
                # Ensure the story has a created_at timestamp
                if not story.get("created_at"):
                    story["created_at"] = datetime.datetime.now().isoformat()
                    
                filename = f"{story['title'].replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                story["filename"] = filename
            
            filepath = self.stories_dir / story["filename"]
            
            with open(filepath, 'w') as f:
                json.dump(story, f, indent=4)
                
            return True
        except Exception as e:
            st.error(f"Error saving story: {str(e)}")
            return False
    
    def load_stories(self):
        """
        Load all stories from the stories directory.
        
        Returns:
            list: List of story dictionaries
        """
        stories = []
        try:
            for file in self.stories_dir.glob("*.json"):
                with open(file, 'r') as f:
                    story = json.load(f)
                    story["filename"] = file.name
                    stories.append(story)
            # Sort stories by creation date (newest first)
            stories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        except Exception as e:
            st.error(f"Error loading stories: {str(e)}")
        return stories
    
    def load_story(self, filename):
        """
        Load a specific story by filename.
        
        Args:
            filename (str): Name of the story file
            
        Returns:
            dict: Story data or None if error
        """
        try:
            filepath = self.stories_dir / filename
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading story: {str(e)}")
            return None
    
    def delete_story(self, filename):
        """
        Delete a story file.
        
        Args:
            filename (str): Name of the story file
            
        Returns:
            bool: Success status
        """
        try:
            filepath = self.stories_dir / filename
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            st.error(f"Error deleting story: {str(e)}")
            return False
    
    def create_new_story(self):
        """
        Create a new empty story structure.
        
        Returns:
            dict: Empty story template
        """
        return {
            "title": "",
            "content": "",
            "chunks": [],
            "created_at": datetime.datetime.now().isoformat(),
            "model": None  # Track which model generated the story
        }