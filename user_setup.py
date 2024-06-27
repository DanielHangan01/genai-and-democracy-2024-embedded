# This file is used to setup the project. It is executed when the project is imported.
# This file should be used to download all large files (e.g., model weights) and store them to disk.
# In this file, you can also check if the environment works as expected.
# If something goes wrong, you can exit the script with a non-zero exit code.
# This will help you detect issues early on.
#
# Below, you can find some sample code:

from sentence_transformers import SentenceTransformer
from user_config import HUGGING_FACE_AUTH
import pkg_resources

def download_large_files():
    try:
        model = SentenceTransformer("mustozsarac/finetuned-four-epoch-multi-qa-mpnet-base-dot-v1", token=HUGGING_FACE_AUTH)
        model.save("./finetuned-four-epoch-multi-qa-mpnet-base-dot-v1/")
        print("Model downloaded and saved successfully.")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def check_environment():
    try:
        with open('requirements.txt') as f:
            requirements = f.read().splitlines()
        
        pkg_resources.require(requirements)
        print("All required packages are installed.")
        return True
    except FileNotFoundError:
        print("Error: requirements.txt file not found.")
        return False
    except pkg_resources.DistributionNotFound as e:
        print(f"Error: {e}")
        return False
    except pkg_resources.VersionConflict as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Performing setup:")
    
    if not check_environment():
        print("Environment check failed.")
        exit(1)
        
    if not download_large_files():
        print("Downloading large files failed.")
        exit(1)
        
    print("Setup completed successfully.")
    exit(0)
