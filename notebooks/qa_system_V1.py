# Question-Answering System using HuggingFace Datasets
# This system loads Q&A datasets, implements similarity search, and answers questions

import pandas as pd
import numpy as np
from datasets import load_dataset, list_datasets
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import warnings
warnings.filterwarnings('ignore')

class QASystem:
    def __init__(self):
        self.qa_data = None
        self.questions = []
        self.answers = []
        self.question_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.embedding_model = None
        self.tokenizer = None
        
    def load_qa_dataset(self, dataset_name="squad"):
        """
        Load Q&A dataset from HuggingFace
        Popular Q&A datasets: 'squad', 'ms_marco', 'natural_questions', 'quac'
        """
        print(f"Loading {dataset_name} dataset...")
        try:
            # Load SQuAD dataset (most popular Q&A dataset)
            if dataset_name == "squad":
                dataset = load_dataset("squad", split="train[:5000]")  # Load first 5000 for demo
            elif dataset_name == "ms_marco":
                dataset = load_dataset("ms_marco", "v1.1", split="train[:5000]")
            else:
                dataset = load_dataset(dataset_name, split="train[:5000]")
            
            # Convert to pandas for easier handling
            self.qa_data = dataset.to_pandas()
            
            # Extract questions and answers based on dataset structure
            if 'question' in self.qa_data.columns:
                self.questions = self.qa_data['question'].tolist()
            elif 'query' in self.qa_data.columns:
                self.questions = self.qa_data['query'].tolist()
                
            if 'answers' in self.qa_data.columns:
                # SQuAD format - answers is a dict with 'text' list
                self.answers = [ans['text'][0] if ans['text'] else "No answer" 
                              for ans in self.qa_data['answers']]
            elif 'answer' in self.qa_data.columns:
                self.answers = self.qa_data['answer'].tolist()
            elif 'passages' in self.qa_data.columns:
                # MS MARCO format
                self.answers = [passage['passage_text'][0] if passage['passage_text'] else "No answer"
                              for passage in self.qa_data['passages']]
            
            print(f"Loaded {len(self.questions)} Q&A pairs")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def setup_embedding_model(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Setup sentence embedding model for semantic similarity"""
        print("Setting up embedding model...")
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
            print("Sentence-transformers model loaded successfully")
        except ImportError:
            print("sentence-transformers not installed, using HuggingFace transformers")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)
    
    def create_embeddings(self):
        """Create embeddings for all questions in the dataset"""
        print("Creating question embeddings...")
        
        if hasattr(self.embedding_model, 'encode'):
            # Using sentence-transformers
            self.question_embeddings = self.embedding_model.encode(self.questions)
        else:
            # Using transformers
            embeddings = []
            for question in self.questions:
                inputs = self.tokenizer(question, return_tensors='pt', truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    embeddings.append(embedding)
            self.question_embeddings = np.array(embeddings)
        
        print(f"Created embeddings for {len(self.questions)} questions")
    
    def setup_tfidf(self):
        """Setup TF-IDF vectorizer for keyword-based similarity"""
        print("Setting up TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.questions)
        print("TF-IDF setup complete")
    
    def find_similar_questions_embedding(self, query, top_k=5):
        """Find similar questions using semantic embeddings"""
        if hasattr(self.embedding_model, 'encode'):
            query_embedding = self.embedding_model.encode([query])
        else:
            inputs = self.tokenizer(query, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        
        # Get top-k most similar questions
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'similarity': similarities[idx],
                'method': 'embedding'
            })
        
        return results
    
    def find_similar_questions_tfidf(self, query, top_k=5):
        """Find similar questions using TF-IDF"""
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k most similar questions
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'similarity': similarities[idx],
                'method': 'tfidf'
            })
        
        return results
    
    def hybrid_search(self, query, top_k=5, embedding_weight=0.7, tfidf_weight=0.3):
        """Combine embedding and TF-IDF results for better accuracy"""
        embedding_results = self.find_similar_questions_embedding(query, top_k*2)
        tfidf_results = self.find_similar_questions_tfidf(query, top_k*2)
        
        # Combine and re-rank results
        combined_scores = {}
        
        for result in embedding_results:
            question = result['question']
            combined_scores[question] = {
                'embedding_score': result['similarity'] * embedding_weight,
                'tfidf_score': 0,
                'answer': result['answer']
            }
        
        for result in tfidf_results:
            question = result['question']
            if question in combined_scores:
                combined_scores[question]['tfidf_score'] = result['similarity'] * tfidf_weight
            else:
                combined_scores[question] = {
                    'embedding_score': 0,
                    'tfidf_score': result['similarity'] * tfidf_weight,
                    'answer': result['answer']
                }
        
        # Calculate final scores and sort
        final_results = []
        for question, scores in combined_scores.items():
            final_score = scores['embedding_score'] + scores['tfidf_score']
            final_results.append({
                'question': question,
                'answer': scores['answer'],
                'similarity': final_score,
                'method': 'hybrid'
            })
        
        # Sort by final score and return top-k
        final_results.sort(key=lambda x: x['similarity'], reverse=True)
        return final_results[:top_k]
    
    def answer_question(self, query, method='hybrid', top_k=3):
        """Main function to answer a question"""
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        if method == 'embedding':
            results = self.find_similar_questions_embedding(query, top_k)
        elif method == 'tfidf':
            results = self.find_similar_questions_tfidf(query, top_k)
        else:
            results = self.hybrid_search(query, top_k)
        
        print(f"Top {len(results)} similar questions and answers:")
        print()
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Question: {result['question']}")
            print(f"   Answer: {result['answer']}")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Method: {result['method']}")
            print()
        
        # Return the best answer
        if results:
            return results[0]['answer']
        else:
            return "I couldn't find a relevant answer in the database."

# Example usage and testing
def main():
    # Initialize the QA system
    qa_system = QASystem()
    
    # Load dataset
    print("Available Q&A datasets:")
    print("1. squad (Stanford Question Answering Dataset)")
    print("2. ms_marco (Microsoft MARCO)")
    print("3. natural_questions")
    
    # Load SQuAD dataset
    if qa_system.load_qa_dataset("squad"):
        
        # Setup both embedding and TF-IDF methods
        qa_system.setup_embedding_model()
        qa_system.create_embeddings()
        qa_system.setup_tfidf()
        
        print("\n" + "="*60)
        print("Q&A SYSTEM READY!")
        print("="*60)
        
        # Test questions
        test_questions = [
            "What is machine learning?",
            "How does artificial intelligence work?",
            "What are neural networks?",
            "Explain deep learning",
            "What is natural language processing?"
        ]
        
        for question in test_questions:
            answer = qa_system.answer_question(question, method='hybrid')
            
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("You can now ask questions! Type 'quit' to exit.")
        
        while True:
            user_question = input("\nYour question: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                break
            if user_question:
                qa_system.answer_question(user_question, method='hybrid')

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import sentence_transformers
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.check_call(["pip", "install", "sentence-transformers"])
    
    main()