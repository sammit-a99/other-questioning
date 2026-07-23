import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_sentences(sentences):
    """Encode sentences into embeddings using SBERT."""
    embeddings = model.encode(sentences)
    return embeddings

def compute_similarity_matrix(embeddings):
    """Compute cosine similarity matrix for embeddings."""
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def analyze_group_similarity(df, text_column='Generated_Questions', group_column='Demographic'):
    """
    Analyze semantic similarity of questions across demographic groups.
    
    Args:
        df: DataFrame with questions and demographic labels
        text_column: Column name containing the questions
        group_column: Column name containing demographic groups
    
    Returns:
        Dictionary with similarity metrics by group
    """
    results = {}
    
    # Get unique groups
    groups = df[group_column].unique()
    
    for group in groups:
        # Get questions for this group
        group_questions = df[df[group_column] == group][text_column].tolist()
        
        if len(group_questions) < 2:
            results[group] = {
                'avg_internal_similarity': None,
                'avg_similarity_to_control': None,
                'sample_questions': group_questions
            }
            continue
        
        # Encode questions
        embeddings = encode_sentences(group_questions)
        
        # Compute internal similarity (within group)
        similarity_matrix = compute_similarity_matrix(embeddings)
        # Get upper triangle (excluding diagonal) for average
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        avg_internal_similarity = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0
        
        results[group] = {
            'avg_internal_similarity': avg_internal_similarity,
            'sample_questions': group_questions[:3]  # Show first 3 questions
        }
    
    # Compute cross-group similarities
    if 'Control' in groups:
        control_questions = df[df[group_column] == 'Control'][text_column].tolist()
        if len(control_questions) > 0:
            control_embeddings = encode_sentences(control_questions)
            
            for group in groups:
                if group == 'Control':
                    continue
                
                group_questions = df[df[group_column] == group][text_column].tolist()
                if len(group_questions) > 0:
                    group_embeddings = encode_sentences(group_questions)
                    
                    # Compute similarity between this group and control
                    cross_similarity = cosine_similarity(group_embeddings, control_embeddings)
                    avg_cross_similarity = np.mean(cross_similarity)
                    
                    results[group]['avg_similarity_to_control'] = avg_cross_similarity
    
    return results

def print_similarity_results(results):
    """Print similarity analysis results."""
    print("\n" + "="*60)
    print("SBERT Semantic Similarity Analysis")
    print("="*60)
    
    for group, metrics in results.items():
        print(f"\n{group}:")
        print(f"  Internal Similarity: {metrics['avg_internal_similarity']:.4f}" if metrics['avg_internal_similarity'] is not None else "  Internal Similarity: N/A")
        if 'avg_similarity_to_control' in metrics and metrics['avg_similarity_to_control'] is not None:
            print(f"  Similarity to Control: {metrics['avg_similarity_to_control']:.4f}")
        print(f"  Sample Questions:")
        for i, q in enumerate(metrics['sample_questions'], 1):
            print(f"    {i}. {q[:100]}...")

def compare_two_questions(question1, question2):
    """
    Compare semantic similarity between two specific questions.
    
    Args:
        question1: First question text
        question2: Second question text
    
    Returns:
        Similarity score (0-1)
    """
    embeddings = encode_sentences([question1, question2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def find_most_similar_pairs(df, text_column='Generated_Questions', group_column='Demographic', top_n=5):
    """
    Find the most similar question pairs across all groups.
    
    Args:
        df: DataFrame with questions
        text_column: Column name containing questions
        group_column: Column name containing demographic groups
        top_n: Number of top similar pairs to return
    
    Returns:
        List of tuples with (similarity, question1, group1, question2, group2)
    """
    # Encode all questions
    all_questions = df[text_column].tolist()
    all_groups = df[group_column].tolist()
    all_embeddings = encode_sentences(all_questions)
    
    # Compute all pairwise similarities
    similarity_matrix = compute_similarity_matrix(all_embeddings)
    
    # Get top similar pairs (excluding self-similarities)
    pairs = []
    for i in range(len(all_questions)):
        for j in range(i+1, len(all_questions)):
            similarity = similarity_matrix[i][j]
            pairs.append((similarity, all_questions[i], all_groups[i], all_questions[j], all_groups[j]))
    
    # Sort by similarity and return top_n
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:top_n]

if __name__ == "__main__":
    # Example usage with sample data
    sample_data = {
        'Demographic': ['Control', 'Control', 'Man', 'Man', 'Woman', 'Woman'],
        'Generated_Questions': [
            "What programming languages do you know?",
            "Tell me about your coding experience.",
            "What languages can you code in?",
            "Describe your software development background.",
            "How do you handle technical challenges?",
            "What's your approach to problem-solving?"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Sample Data:")
    print(df)
    
    # Run similarity analysis
    results = analyze_group_similarity(df)
    print_similarity_results(results)
    
    # Find most similar pairs
    print("\n" + "="*60)
    print("Most Similar Question Pairs")
    print("="*60)
    similar_pairs = find_most_similar_pairs(df, top_n=3)
    for i, (sim, q1, g1, q2, g2) in enumerate(similar_pairs, 1):
        print(f"\n{i}. Similarity: {sim:.4f}")
        print(f"   {g1}: {q1}")
        print(f"   {g2}: {q2}")
