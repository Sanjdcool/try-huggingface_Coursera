# Advanced Intent-Based Q&A System with Academic Support
# Enhanced system with proper error handling and dependency management

import pandas as pd
import numpy as np
import torch
import re
import warnings
import sys
import subprocess
warnings.filterwarnings('ignore')

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        'datasets': 'datasets>=2.0.0',
        'transformers': 'transformers>=4.20.0',
        'sentence-transformers': 'sentence-transformers>=2.2.0',
        'scikit-learn': 'scikit-learn>=1.0.0',
        'accelerate': 'accelerate>=0.20.0'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            if not install_package(package):
                print(f"Failed to install {package}. Please install manually.")
                return False
    
    return True

class AdvancedQASystem:
    def __init__(self):
        self.qa_data = None
        self.questions = []
        self.answers = []
        self.contexts = []
        self.question_types = []
        self.academic_subjects = []
        
        # Models and embeddings
        self.question_embeddings = None
        self.context_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.embedding_model = None
        self.use_sentence_transformers = False
        
        # Intent and category mappings
        self.intent_categories = {
            'definition': ['what is', 'define', 'meaning of', 'explain'],
            'how_to': ['how to', 'how do', 'how can', 'steps to'],
            'comparison': ['difference between', 'compare', 'vs', 'versus'],
            'factual': ['when', 'where', 'who', 'which'],
            'causal': ['why', 'because', 'reason', 'cause'],
            'academic': ['theorem', 'proof', 'formula', 'equation', 'principle'],
            'practical': ['example', 'application', 'use case', 'implement']
        }
        
        # Fixed: Changed from nested dictionary to simple dictionary
        self.academic_subject_keywords = {
            'mathematics': ['math', 'algebra', 'calculus', 'geometry', 'statistics', 'probability'],
            'physics': ['physics', 'mechanics', 'thermodynamics', 'quantum', 'relativity'],
            'chemistry': ['chemistry', 'organic', 'inorganic', 'biochemistry', 'molecular'],
            'computer_science': ['programming', 'algorithm', 'data structure', 'software', 'coding'],
            'biology': ['biology', 'genetics', 'evolution', 'cell', 'organism'],
            'engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'chemical']
        }
    
    def load_multiple_datasets(self):
        """Load multiple Q&A datasets with better error handling"""
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets library...")
            if not install_package('datasets>=2.0.0'):
                print("Failed to install datasets. Using fallback data.")
                return self.create_fallback_data()
            from datasets import load_dataset
        
        datasets_info = [
            {'name': 'squad', 'type': 'general', 'samples': 1000},  # Reduced for faster loading
        ]
        
        all_questions = []
        all_answers = []
        all_contexts = []
        all_types = []
        all_subjects = []
        
        for dataset_info in datasets_info:
            print(f"Loading {dataset_info['name']} dataset...")
            try:
                if dataset_info['name'] == 'squad':
                    dataset = load_dataset("squad", split=f"train[:{dataset_info['samples']}]", trust_remote_code=True)
                    df = dataset.to_pandas()
                    
                    questions = df['question'].tolist()
                    answers = [ans['text'][0] if ans['text'] else "No answer" for ans in df['answers']]
                    contexts = df['context'].tolist()
                    types = [dataset_info['type']] * len(questions)
                    subjects = ['general'] * len(questions)
                    
                    all_questions.extend(questions)
                    all_answers.extend(answers)
                    all_contexts.extend(contexts)
                    all_types.extend(types)
                    all_subjects.extend(subjects)
                    
                    print(f"Loaded {len(questions)} questions from {dataset_info['name']}")
                
            except Exception as e:
                print(f"Error loading {dataset_info['name']}: {e}")
                print("Using fallback data instead...")
                return self.create_fallback_data()
        
        if not all_questions:
            print("No datasets loaded successfully. Using fallback data.")
            return self.create_fallback_data()
        
        self.questions = all_questions
        self.answers = all_answers
        self.contexts = all_contexts
        self.question_types = all_types
        self.academic_subjects = all_subjects
        
        print(f"Total loaded: {len(self.questions)} Q&A pairs")
        return True
    
    def create_fallback_data(self):
        """Create fallback Q&A data when datasets can't be loaded"""
        fallback_qa = [
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                "context": "Machine learning algorithms build mathematical models based on training data to make predictions or decisions.",
                "type": "general",
                "subject": "computer_science"
            },
            {
                "question": "How do neural networks work?",
                "answer": "Neural networks work by processing information through interconnected nodes (neurons) that learn patterns in data through training.",
                "context": "Artificial neural networks are inspired by biological neural networks and use layers of connected nodes.",
                "type": "general",
                "subject": "computer_science"
            },
            {
                "question": "What is the Pythagorean theorem?",
                "answer": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c².",
                "context": "This fundamental theorem in geometry is named after the ancient Greek mathematician Pythagoras.",
                "type": "academic",
                "subject": "mathematics"
            },
            {
                "question": "What is photosynthesis?",
                "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
                "context": "This process occurs in chloroplasts and is essential for life on Earth as it produces oxygen and food.",
                "type": "academic",
                "subject": "biology"
            },
            {
                "question": "What is Newton's first law?",
                "answer": "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by an external force.",
                "context": "Also known as the law of inertia, this is one of the fundamental laws of classical mechanics.",
                "type": "academic",
                "subject": "physics"
            }
        ]
        
        self.questions = [qa["question"] for qa in fallback_qa]
        self.answers = [qa["answer"] for qa in fallback_qa]
        self.contexts = [qa["context"] for qa in fallback_qa]
        self.question_types = [qa["type"] for qa in fallback_qa]
        self.academic_subjects = [qa["subject"] for qa in fallback_qa]
        
        print(f"Created fallback dataset with {len(self.questions)} Q&A pairs")
        return True
    
    def classify_intent(self, question):
        """Classify the intent of a question"""
        question_lower = question.lower()
        
        for intent, keywords in self.intent_categories.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return intent
        
        # Use pattern matching for more complex intent detection
        if re.search(r'\b(what|how|why|when|where|who|which)\b', question_lower):
            if 'what' in question_lower:
                if any(word in question_lower for word in ['is', 'are', 'mean', 'define']):
                    return 'definition'
                else:
                    return 'factual'
            elif 'how' in question_lower:
                return 'how_to'
            elif 'why' in question_lower:
                return 'causal'
            else:
                return 'factual'
        
        return 'general'
    
    def classify_academic_subject(self, question):
        """Classify the academic subject of a question"""
        question_lower = question.lower()
        
        subject_scores = {}
        # Fixed: Use the correct attribute name
        for subject, keywords in self.academic_subject_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                subject_scores[subject] = score
        
        if subject_scores:
            return max(subject_scores, key=subject_scores.get)
        return 'general'
    
    def setup_advanced_models(self):
        """Setup advanced models with fallback options"""
        print("Setting up advanced models...")
        
        # Try to setup sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            self.use_sentence_transformers = True
            print("✓ Sentence transformer model loaded")
        except Exception as e:
            print(f"Sentence transformers not available: {e}")
            print("✓ Using TF-IDF as fallback")
            self.use_sentence_transformers = False
        
        # Setup TF-IDF (always available)
        self.setup_tfidf()
        print("✓ TF-IDF vectorizer setup complete")
    
    def create_advanced_embeddings(self):
        """Create embeddings with fallback to TF-IDF"""
        if self.use_sentence_transformers:
            print("Creating sentence transformer embeddings...")
            try:
                self.question_embeddings = self.embedding_model.encode(
                    self.questions, 
                    show_progress_bar=True,
                    batch_size=32
                )
                
                self.context_embeddings = self.embedding_model.encode(
                    self.contexts,
                    show_progress_bar=True,
                    batch_size=32
                )
                print(f"✓ Created embeddings for {len(self.questions)} questions and contexts")
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                print("Falling back to TF-IDF only")
                self.use_sentence_transformers = False
        else:
            print("✓ Using TF-IDF for similarity matching")
    
    def setup_tfidf(self):
        """Setup enhanced TF-IDF vectorizer"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Reduced for better performance
            stop_words='english',
            ngram_range=(1, 2),  # Reduced to bigrams
            min_df=1,  # Allow single occurrences for small datasets
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Combine questions and contexts for better TF-IDF
        combined_text = [f"{q} {c}" for q, c in zip(self.questions, self.contexts)]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
    
    def intent_aware_search(self, query, top_k=5):
        """Search based on intent classification with fallback methods"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_intent = self.classify_intent(query)
        query_subject = self.classify_academic_subject(query)
        
        print(f"Detected Intent: {query_intent}")
        print(f"Detected Subject: {query_subject}")
        
        # Get TF-IDF similarity (always available)
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        if self.use_sentence_transformers:
            # Get semantic similarity
            query_embedding = self.embedding_model.encode([query])
            question_similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
            context_similarities = cosine_similarity(query_embedding, self.context_embeddings)[0]
            
            # Combine question and context similarities
            combined_similarities = 0.7 * question_similarities + 0.3 * context_similarities
            
            # Final scoring with semantic + TF-IDF
            final_scores = 0.6 * combined_similarities + 0.4 * tfidf_similarities
        else:
            # Use only TF-IDF
            final_scores = tfidf_similarities
        
        # Intent-based boosting
        intent_boost = np.ones(len(self.questions))
        for i, question in enumerate(self.questions):
            question_intent = self.classify_intent(question)
            question_subject = self.classify_academic_subject(question)
            
            # Boost if intents match
            if question_intent == query_intent:
                intent_boost[i] *= 1.5
            
            # Boost if subjects match
            if question_subject == query_subject and query_subject != 'general':
                intent_boost[i] *= 1.3
        
        # Apply intent boosting
        final_scores = final_scores * intent_boost
        
        # Get top results
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'context': self.contexts[idx][:200] + "..." if len(self.contexts[idx]) > 200 else self.contexts[idx],
                'similarity': final_scores[idx],
                'intent': self.classify_intent(self.questions[idx]),
                'subject': self.classify_academic_subject(self.questions[idx]),
                'question_type': self.question_types[idx],
                'method': 'intent_aware'
            })
        
        return results
    
    def answer_question(self, query, method='intent_aware', top_k=3):
        """Enhanced main function to answer questions"""
        print(f"\n🤖 Query: {query}")
        print("-" * 80)
        
        results = self.intent_aware_search(query, top_k)
        
        if not results:
            return "I couldn't find a relevant answer in the database."
        
        print(f"🎯 Top {len(results)} results:")
        print()
        
        for i, result in enumerate(results, 1):
            print(f"📌 {i}. Intent: {result['intent'].upper()} | Subject: {result['subject'].upper()}")
            print(f"❓ Question: {result['question']}")
            print(f"💡 Answer: {result['answer']}")
            if result['context'] and len(result['context']) > 10:
                print(f"📚 Context: {result['context']}")
            print(f"🎲 Confidence: {result['similarity']:.4f}")
            print("-" * 40)
        
        return results[0]['answer']
    
    def get_statistics(self):
        """Get system statistics"""
        if not self.questions:
            return "No data loaded"
        
        intent_counts = {}
        subject_counts = {}
        
        for question in self.questions:
            intent = self.classify_intent(question)
            subject = self.classify_academic_subject(question)
            
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        
        print("\n📊 System Statistics:")
        print(f"Total Questions: {len(self.questions)}")
        print(f"Using Sentence Transformers: {self.use_sentence_transformers}")
        print(f"Intent Distribution: {dict(sorted(intent_counts.items(), key=lambda x: x[1], reverse=True))}")
        print(f"Subject Distribution: {dict(sorted(subject_counts.items(), key=lambda x: x[1], reverse=True))}")

def main():
    print("🚀 Initializing Advanced Intent-Based Q&A System...")
    print("="*80)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_and_install_dependencies():
        print("Some dependencies are missing. The system will use fallback methods.")
    
    # Initialize the advanced QA system
    qa_system = AdvancedQASystem()
    
    # Load datasets
    if qa_system.load_multiple_datasets():
        
        # Setup models
        qa_system.setup_advanced_models()
        qa_system.create_advanced_embeddings()
        
        # Show system statistics
        qa_system.get_statistics()
        
        print("\n" + "="*80)
        print("🎉 ADVANCED Q&A SYSTEM READY!")
        print("="*80)
        
        # Test with different types of questions
        test_questions = [
            "What is machine learning?",
            "How do neural networks work?",
            "What is the Pythagorean theorem?",
            "Explain photosynthesis process"
        ]
        
        print("\n🧪 Testing with sample questions:")
        for question in test_questions:
            qa_system.answer_question(question, top_k=2)
        
        print("\n" + "="*80)
        print("💬 INTERACTIVE MODE:")
        print("Commands:")
        print("  • Type your question")
        print("  • Type 'stats' for system statistics")
        print("  • Type 'quit' to exit")
        print("="*80)
        
        while True:
            user_input = input("\n🎤 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'stats':
                qa_system.get_statistics()
                continue
            elif not user_input:
                continue
            
            # Get answer
            qa_system.answer_question(user_input, top_k=3)
    
    else:
        print("❌ Failed to initialize the system. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()