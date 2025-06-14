import nltk
import urllib.request
import ssl

# For corporate networks or firewalls
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('stopwords')