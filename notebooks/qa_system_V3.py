# Advanced Intent-Based Q&A System WITHOUT NLTK
# This version avoids NLTK completely to prevent import errors

import pandas as pd
import numpy as np
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

class SimpleQASystem:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.contexts = []
        self.question_types = []
        self.question_subjects = []  # Changed from academic_subjects to avoid confusion
        
        # Models and embeddings
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.use_transformers = False
        
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
        
        self.subject_keywords = {  # Renamed for clarity
            'mathematics': ['math', 'algebra', 'calculus', 'geometry', 'statistics', 'probability'],
            'physics': ['physics', 'mechanics', 'thermodynamics', 'quantum', 'relativity'],
            'chemistry': ['chemistry', 'organic', 'inorganic', 'biochemistry', 'molecular'],
            'computer_science': ['programming', 'algorithm', 'data structure', 'software', 'coding'],
            'biology': ['biology', 'genetics', 'evolution', 'cell', 'organism'],
            'engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'chemical']
        }
    
    def create_comprehensive_fallback_data(self):
        """Create comprehensive fallback Q&A data"""
        fallback_qa = [
            # Computer Science
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
                "context": "Machine learning algorithms build mathematical models based on training data to make predictions or decisions on new data.",
                "type": "general",
                "subject": "computer_science"
            },
            {
                "question": "How do neural networks work?",
                "answer": "Neural networks work by processing information through interconnected nodes (neurons) organized in layers. Each connection has a weight, and the network learns by adjusting these weights through training.",
                "context": "Artificial neural networks are inspired by biological neural networks and use mathematical functions to process and transform input data.",
                "type": "general",
                "subject": "computer_science"
            },
            {
                "question": "What is an algorithm?",
                "answer": "An algorithm is a step-by-step set of instructions or rules designed to solve a specific problem or perform a particular task.",
                "context": "Algorithms are fundamental to computer science and programming, providing systematic approaches to problem-solving.",
                "type": "general",
                "subject": "computer_science"
            },
            
            # Mathematics
            {
                "question": "What is the Pythagorean theorem?",
                "answer": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c².",
                "context": "This fundamental theorem in geometry is named after the ancient Greek mathematician Pythagoras and is used extensively in mathematics and engineering.",
                "type": "academic",
                "subject": "mathematics"
            },
            {
                "question": "What is calculus?",
                "answer": "Calculus is a branch of mathematics that deals with rates of change (differential calculus) and accumulation of quantities (integral calculus).",
                "context": "Calculus was developed independently by Newton and Leibniz and is essential for physics, engineering, and many other fields.",
                "type": "academic",
                "subject": "mathematics"
            },
            
            # Physics
            {
                "question": "What is Newton's first law?",
                "answer": "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with constant velocity unless acted upon by an external force.",
                "context": "Also known as the law of inertia, this is one of the fundamental laws of classical mechanics.",
                "type": "academic",
                "subject": "physics"
            },
            {
                "question": "What is quantum mechanics?",
                "answer": "Quantum mechanics is the branch of physics that describes the behavior of matter and energy at the atomic and subatomic scale.",
                "context": "Quantum mechanics reveals that particles exhibit both wave-like and particle-like properties and introduces concepts like uncertainty and superposition.",
                "type": "academic",
                "subject": "physics"
            },
            
            # Biology
            {
                "question": "What is photosynthesis?",
                "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
                "context": "This process occurs in chloroplasts and is essential for life on Earth as it produces oxygen and serves as the base of most food chains.",
                "type": "academic",
                "subject": "biology"
            },
            {
                "question": "What is DNA?",
                "answer": "DNA (Deoxyribonucleic acid) is the hereditary material that contains genetic instructions for the development and function of living organisms.",
                "context": "DNA has a double-helix structure and is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
                "type": "academic",
                "subject": "biology"
            },
            
            # Chemistry
            {
                "question": "What is an atom?",
                "answer": "An atom is the smallest unit of matter that retains the properties of an element, consisting of a nucleus (protons and neutrons) surrounded by electrons.",
                "context": "Atoms are the building blocks of all matter and combine to form molecules and compounds.",
                "type": "academic",
                "subject": "chemistry"
            },
            
            # General Knowledge
            {
                "question": "How does the internet work?",
                "answer": "The internet works through a global network of interconnected computers that communicate using standardized protocols like TCP/IP to exchange data.",
                "context": "The internet relies on routers, switches, and servers to route data packets between devices worldwide.",
                "type": "general",
                "subject": "computer_science"
            },
            {
                "question": "What is artificial intelligence?",
                "answer": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think, learn, and make decisions like humans.",
                "context": "AI encompasses various technologies including machine learning, natural language processing, and computer vision.",
                "type": "general",
                "subject": "computer_science"
            }
        ]
        
        self.questions = [qa["question"] for qa in fallback_qa]
        self.answers = [qa["answer"] for qa in fallback_qa]
        self.contexts = [qa["context"] for qa in fallback_qa]
        self.question_types = [qa["type"] for qa in fallback_qa]
        self.question_subjects = [qa["subject"] for qa in fallback_qa]  # Fixed variable name
        
        print(f"✓ Created comprehensive dataset with {len(self.questions)} Q&A pairs")
        return True
    
    def load_datasets_safely(self):
        """Try to load datasets, fall back to local data if needed"""
        print("Attempting to load external datasets...")
        
        try:
            from datasets import load_dataset
            print("✓ Datasets library available")
            
            # Try to load a small sample
            dataset = load_dataset("squad", split="train[:100]", trust_remote_code=True)
            df = dataset.to_pandas()
            
            questions = df['question'].tolist()
            answers = [ans['text'][0] if ans['text'] else "No answer" for ans in df['answers']]
            contexts = df['context'].tolist()
            types = ['general'] * len(questions)
            subjects = ['general'] * len(questions)
            
            # Add to existing data
            self.questions.extend(questions)
            self.answers.extend(answers)
            self.contexts.extend(contexts)
            self.question_types.extend(types)
            self.question_subjects.extend(subjects)  # Fixed variable name
            
            print(f"✓ Added {len(questions)} questions from SQuAD dataset")
            
        except Exception as e:
            print(f"Could not load external datasets: {e}")
            print("✓ Using local data only")
        
        return True
    
    def classify_intent(self, question):
        """Classify the intent of a question"""
        question_lower = question.lower()
        
        for intent, keywords in self.intent_categories.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return intent
        
        # Pattern matching
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
        for subject, keywords in self.subject_keywords.items():  # Fixed variable name
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                subject_scores[subject] = score
        
        if subject_scores:
            return max(subject_scores, key=subject_scores.get)
        return 'general'
    
    def setup_tfidf(self):
        """Setup TF-IDF vectorizer"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=3000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )
            
            # Combine questions and contexts
            combined_text = [f"{q} {c}" for q, c in zip(self.questions, self.contexts)]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
            
            print("✓ TF-IDF vectorizer setup complete")
            return True
            
        except ImportError:
            print("scikit-learn not available. Installing...")
            if install_package('scikit-learn'):
                return self.setup_tfidf()
            else:
                print("✗ Could not setup TF-IDF. Using simple text matching.")
                return False
    
    def simple_text_similarity(self, query, text):
        """Simple text similarity without external libraries"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        # Jaccard similarity
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def search_questions(self, query, top_k=5):
        """Search for relevant questions"""
        query_intent = self.classify_intent(query)
        query_subject = self.classify_academic_subject(query)
        
        print(f"Detected Intent: {query_intent}")
        print(f"Detected Subject: {query_subject}")
        
        scores = []
        
        if hasattr(self, 'tfidf_matrix') and self.tfidf_matrix is not None:
            # Use TF-IDF if available
            from sklearn.metrics.pairwise import cosine_similarity
            
            query_tfidf = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            
            for i, sim in enumerate(similarities):
                # Intent and subject boosting
                boost = 1.0
                question_intent = self.classify_intent(self.questions[i])
                question_subject = self.classify_academic_subject(self.questions[i])
                
                if question_intent == query_intent:
                    boost *= 1.5
                if question_subject == query_subject and query_subject != 'general':
                    boost *= 1.3
                
                scores.append((i, sim * boost))
        else:
            # Fallback to simple similarity
            for i, question in enumerate(self.questions):
                sim = self.simple_text_similarity(query, f"{question} {self.contexts[i]}")
                
                # Intent and subject boosting
                boost = 1.0
                question_intent = self.classify_intent(question)
                question_subject = self.classify_academic_subject(question)
                
                if question_intent == query_intent:
                    boost *= 1.5
                if question_subject == query_subject and query_subject != 'general':
                    boost *= 1.3
                
                scores.append((i, sim * boost))
        
        # Sort by score and get top results
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in scores[:top_k]]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'context': self.contexts[idx][:200] + "..." if len(self.contexts[idx]) > 200 else self.contexts[idx],
                'similarity': scores[i][1],
                'intent': self.classify_intent(self.questions[idx]),
                'subject': self.classify_academic_subject(self.questions[idx]),
                'question_type': self.question_types[idx]
            })
        
        return results
    
    def answer_question(self, query, top_k=3):
        """Answer a question"""
        print(f"\n🤖 Query: {query}")
        print("-" * 80)
        
        results = self.search_questions(query, top_k)
        
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
        print(f"TF-IDF Available: {hasattr(self, 'tfidf_matrix') and self.tfidf_matrix is not None}")
        print(f"Intent Distribution: {dict(sorted(intent_counts.items(), key=lambda x: x[1], reverse=True))}")
        print(f"Subject Distribution: {dict(sorted(subject_counts.items(), key=lambda x: x[1], reverse=True))}")

def main():
    print("🚀 Initializing Simple Q&A System (NLTK-Free)...")
    print("="*80)
    
    # Initialize system
    qa_system = SimpleQASystem()
    
    # Create local data first
    qa_system.create_comprehensive_fallback_data()
    
    # Try to load additional datasets
    qa_system.load_datasets_safely()
    
    # Setup TF-IDF
    qa_system.setup_tfidf()
    
    # Show statistics
    qa_system.get_statistics()
    
    print("\n" + "="*80)
    print("🎉 Q&A SYSTEM READY!")
    print("="*80)
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is the Pythagorean theorem?",
        "Explain photosynthesis",
        "What is quantum mechanics?"
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
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'stats':
            qa_system.get_statistics()
            continue
        elif not user_input:
            continue
        
        qa_system.answer_question(user_input, top_k=3)

if __name__ == "__main__":
    main()