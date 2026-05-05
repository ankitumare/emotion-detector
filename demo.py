#!/usr/bin/env python3
"""
Production-grade ML Pipeline Demo Script

This script demonstrates how to use the sentiment analysis model
for real-time predictions.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.predict import SentimentPredictor


def interactive_demo():
    """
    Interactive demo for real-time sentiment prediction.
    """
    print("🤖 Sentiment Analysis Model Demo")
    print("=" * 40)
    
    try:
        # Initialize predictor
        print("Loading model and vectorizer...")
        predictor = SentimentPredictor()
        print("✅ Model loaded successfully!\n")
        
        # Show model info
        model_info = predictor.get_model_info()
        print("Model Information:")
        for key, value in model_info.items():
            print(f"  • {key}: {value}")
        print()
        
        # Interactive prediction
        print("Enter text to analyze sentiment (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                text = input("\nEnter text: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    print("⚠️  Please enter some text.")
                    continue
                
                # Make prediction
                result = predictor.predict_single(text)
                
                # Display results
                print(f"\n📝 Original text: {result['text']}")
                print(f"🔧 Processed text: {result['processed_text']}")
                print(f"😊 Sentiment: {result['sentiment'].upper()}")
                print(f"📊 Confidence: {result['confidence']:.2%}")
                print(f"📈 Probabilities:")
                print(f"   • Sadness: {result['probabilities']['sadness']:.2%}")
                print(f"   • Happiness: {result['probabilities']['happiness']:.2%}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        sys.exit(1)


def batch_demo():
    """
    Batch prediction demo.
    """
    print("📊 Batch Prediction Demo")
    print("=" * 30)
    
    try:
        # Initialize predictor
        predictor = SentimentPredictor()
        
        # Example batch texts
        texts = [
            "I love this product! It's amazing!",
            "This is terrible, I hate it.",
            "It's okay, nothing special.",
            "Best purchase I've ever made!",
            "Not worth the money at all.",
            "Pretty good overall experience.",
            "Absolutely fantastic service!",
            "Could be better, disappointing."
        ]
        
        print(f"Analyzing {len(texts)} texts...\n")
        
        # Make batch predictions
        results = predictor.predict_batch(texts)
        
        # Display results
        print("Batch Results:")
        print("-" * 80)
        
        for result in results:
            if "error" in result:
                print(f"❌ Error at index {result['batch_index']}: {result['error']}")
            else:
                print(f"📝 {result['text'][:40]}{'...' if len(result['text']) > 40 else ''}")
                print(f"   😊 {result['sentiment'].upper()} ({result['confidence']:.2%})")
                print()
        
        # Summary statistics
        successful = sum(1 for r in results if "error" not in r)
        happy_count = sum(1 for r in results if r.get('sentiment') == 'happiness')
        sad_count = sum(1 for r in results if r.get('sentiment') == 'sadness')
        
        print("Summary Statistics:")
        print(f"  • Total texts: {len(texts)}")
        print(f"  • Successful predictions: {successful}")
        print(f"  • Happy sentiments: {happy_count}")
        print(f"  • Sad sentiments: {sad_count}")
        
    except Exception as e:
        print(f"❌ Batch demo failed: {e}")


def main():
    """
    Main demo function.
    """
    print("🚀 Production-Grade ML Pipeline Demo")
    print("=" * 50)
    print("Choose demo mode:")
    print("1. Interactive (real-time predictions)")
    print("2. Batch (multiple texts at once)")
    print("3. Model information only")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            interactive_demo()
        elif choice == '2':
            batch_demo()
        elif choice == '3':
            predictor = SentimentPredictor()
            model_info = predictor.get_model_info()
            print("\nModel Information:")
            for key, value in model_info.items():
                print(f"  • {key}: {value}")
        else:
            print("❌ Invalid choice. Please run again.")
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == '__main__':
    main()
