"""
Model Version Manager for tracking EasyOCR model versions
"""
import json
import os
from pathlib import Path
from loguru import logger


class ModelVersionManager:
    def __init__(self, version_file_path=None):
        """
        Initialize the model version manager
        """
        if version_file_path is None:
            self.version_file_path = Path(os.getenv("MODEL_VERSION_FILE", "./model_versions.json"))
        else:
            self.version_file_path = Path(version_file_path)
        
        # Ensure the file exists with initial version
        self._ensure_version_file_exists()
    
    def _ensure_version_file_exists(self):
        """Ensure the version file exists with initial data"""
        if not self.version_file_path.exists():
            initial_data = {
                "current_version": "v1.0-EasyOCR",
                "version_history": [
                    {
                        "version": "v1.0-EasyOCR",
                        "timestamp": "initial",
                        "description": "Initial EasyOCR model version"
                    }
                ]
            }
            self._save_version_data(initial_data)
    
    def _load_version_data(self):
        """Load version data from file"""
        try:
            with open(self.version_file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Could not load version file {self.version_file_path}, creating default")
            self._ensure_version_file_exists()
            return self._load_version_data()
    
    def _save_version_data(self, data):
        """Save version data to file"""
        with open(self.version_file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_current_version(self):
        """Get the current model version"""
        data = self._load_version_data()
        return data.get("current_version", "v1.0-EasyOCR")
    
    def increment_version(self, description=""):
        """Increment the model version after retraining"""
        data = self._load_version_data()
        current_version = data.get("current_version", "v1.0-EasyOCR")

        # Extract version number and increment it
        try:
            # Parse version like "v1.0-EasyOCR" to get the numeric part
            # Split by "-" to separate version from name, then remove "v" prefix
            version_part = current_version.replace("v", "").split("-")[0]  # Gets "1.0"
            parts = version_part.split(".")

            if len(parts) >= 2:
                major = int(parts[0])
                minor = int(parts[1])

                # Increment minor version
                new_minor = minor + 1
                new_version = f"v{major}.{new_minor}-EasyOCR"
            else:
                # If format is unexpected, default to v1.1
                new_version = f"v1.1-EasyOCR"
        except (ValueError, IndexError):
            # If parsing fails, default to incrementing
            new_version = f"v1.1-EasyOCR"  # Default to 1.1 if parsing fails

        # Update the data
        data["current_version"] = new_version
        data["version_history"].append({
            "version": new_version,
            "timestamp": self._get_timestamp(),
            "description": description or f"Model retrained with new data"
        })

        self._save_version_data(data)
        logger.info(f"Model version incremented to: {new_version}")
        return new_version
    
    def _get_timestamp(self):
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_version_history(self):
        """Get the complete version history"""
        data = self._load_version_data()
        return data.get("version_history", [])


# Global instance for convenience
version_manager = ModelVersionManager()


def get_current_model_version():
    """Get the current model version"""
    return version_manager.get_current_version()


def increment_model_version(description=""):
    """Increment the model version after retraining"""
    return version_manager.increment_version(description)


def get_model_version_history():
    """Get the model version history"""
    return version_manager.get_version_history()


if __name__ == "__main__":
    # Test the version manager
    print("Current version:", get_current_model_version())
    print("Incrementing version...")
    new_version = increment_model_version("Test retraining")
    print("New version:", new_version)
    print("Version history:", get_model_version_history())