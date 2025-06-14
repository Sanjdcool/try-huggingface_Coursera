# import torch
# print(torch.cuda.is_available())  # Should return True if CUDA is enabled
# Run this script first to download required NLTK data
import nltk
import ssl

# Handle SSL certificate issues if they occur
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("✓ NLTK data downloaded successfully!")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print("You may need to run this manually or check your internet connection.")