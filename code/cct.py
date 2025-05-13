"""
Cultural Consensus Theory (CCT) Model Implementation

Script implements the basic CCT model using PyMC to analyze plant knowledge data.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def load_plant_knowledge_data(filepath):
    """
    Load the plant knowledge dataset and prepare it for analysis.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the CSV file containing the plant knowledge data
    
    Returns:
    --------
    numpy.ndarray
        The data matrix (N x M) where N is number of informants, M is number of items
    """
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Remove the Informant ID column and convert to numpy array
        data_matrix = df.drop(columns=['Informant']).values
        
        print(f"Loaded data with shape: {data_matrix.shape}")
        print(f"Number of informants: {data_matrix.shape[0]}")
        print(f"Number of items (questions): {data_matrix.shape[1]}")
        
        return data_matrix
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None



def create_cct_model(data):
    """
    Create the CCT model using PyMC.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The observed data matrix (N x M)
    
    Returns:
    --------
    pm.Model
        The PyMC model
    """
    N, M = data.shape
    
    with pm.Model() as model:
        # Prior for competence (D): Uniform distribution between 0.5 and 1
        # Justification: Competence must be at least 0.5 (better than random),
        # and at most 1 (perfect knowledge). Uniform reflects minimal assumptions.
        D = pm.Uniform('D', lower=0.5, upper=1.0, shape=N)
        
        # Prior for consensus answers (Z): Bernoulli with probability 0.5
        # Justification: This reflects minimal prior assumption about what the
        # correct answers are - each could be 0 or 1 with equal probability.
        Z = pm.Bernoulli('Z', p=0.5, shape=M)
        
        # Reshape D for broadcasting with Z
        # D_reshaped will be (N, 1) to allow proper broadcasting
        D_reshaped = D[:, None]
        
        # Calculate the probability matrix p
        # p[i,j] = probability that informant i answers '1' for question j
        # If Z[j] = 1: p[i,j] = D[i] (probability of correct answer)
        # If Z[j] = 0: p[i,j] = 1 - D[i] (probability of incorrect answer)
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)
        
        # Likelihood: observed responses follow Bernoulli distribution
        observed_responses = pm.Bernoulli('observed_responses', p=p, observed=data)
    
    return model

#chat gpt used to help organize the structure of first section 

def run_inference(model, draws=2000, chains=4, tune=1000):
    """
    Run MCMC inference on the CCT model.
    
    Parameters:
    -----------
    model : pm.Model
        The PyMC model
    draws : int
        Number of samples to draw per chain
    chains : int
        Number of MCMC chains
    tune : int
        Number of tuning steps
    
    Returns:
    --------
    arviz.InferenceData
        The trace containing the MCMC samples
    """
    with model:
        # Sample from the posterior
        trace = pm.sample(draws=draws, chains=chains, tune=tune, 
                         random_seed=42, return_inferencedata=True)
    
    return trace

def analyze_convergence(trace):
    """
    Analyze convergence diagnostics.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        The MCMC trace
    """
    print("=== Convergence Diagnostics ===")
    print(az.summary(trace))
    
    # Check R-hat values
    summary = az.summary(trace)
    max_rhat = summary['r_hat'].max()
    print(f"\nMaximum R-hat value: {max_rhat:.4f}")
    
    if max_rhat < 1.01:
        print("✓ Model appears to have converged (all R-hat < 1.01)")
    else:
        print("⚠ Model may not have converged (some R-hat >= 1.01)")
    
    # Plot pair plot for key variables
    az.plot_pair(trace, var_names=['D', 'Z'], figsize=(12, 10))
    plt.suptitle('Pair Plot of Model Parameters', fontsize=16)
    plt.tight_layout()
    plt.savefig('convergence_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_competence(trace, data):
    """
    Analyze informant competence.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        The MCMC trace
    data : numpy.ndarray
        The original data matrix
    """
    print("\n=== Informant Competence Analysis ===")
    
    # Extract competence estimates
    D_posterior = trace.posterior['D']
    D_mean = D_posterior.mean(dim=['chain', 'draw']).values
    D_std = D_posterior.std(dim=['chain', 'draw']).values
    
    # Create a DataFrame for easy viewing
    competence_df = pd.DataFrame({
        'Informant': range(1, len(D_mean) + 1),
        'Competence_Mean': D_mean,
        'Competence_Std': D_std
    })
    
    # Sort by competence
    competence_df = competence_df.sort_values('Competence_Mean', ascending=False)
    print("\nInformant Competence Estimates (sorted by competence):")
    print(competence_df.to_string(index=False, float_format='%.3f'))
    
    # Identify most and least competent informants
    most_competent = competence_df.iloc[0]
    least_competent = competence_df.iloc[-1]
    
    print(f"\nMost competent informant: {most_competent['Informant']} (competence = {most_competent['Competence_Mean']:.3f})")
    print(f"Least competent informant: {least_competent['Informant']} (competence = {least_competent['Competence_Mean']:.3f})")
    
    # Visualize competence distributions
    plt.figure(figsize=(12, 8))
    az.plot_posterior(trace, var_names=['D'], hdi_prob=0.95)
    plt.suptitle('Posterior Distributions of Informant Competence', fontsize=16)
    plt.tight_layout()
    plt.savefig('competence_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return competence_df

def analyze_consensus(trace, data):
    """
    Analyze consensus answers.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        The MCMC trace
    data : numpy.ndarray
        The original data matrix
    """
    print("\n=== Consensus Answers Analysis ===")
    
    # Extract consensus estimates
    Z_posterior = trace.posterior['Z']
    Z_mean = Z_posterior.mean(dim=['chain', 'draw']).values
    
    # Get consensus answers (round to nearest integer)
    consensus_answers = np.round(Z_mean).astype(int)
    
    # Create a DataFrame for easy viewing
    consensus_df = pd.DataFrame({
        'Question': [f'PQ{i}' for i in range(1, len(Z_mean) + 1)],
        'Consensus_Probability': Z_mean,
        'Consensus_Answer': consensus_answers
    })
    
    print("\nConsensus Answers:")
    print(consensus_df.to_string(index=False, float_format='%.3f'))
    
    # Visualize consensus probabilities
    plt.figure(figsize=(12, 8))
    az.plot_posterior(trace, var_names=['Z'], hdi_prob=0.95)
    plt.suptitle('Posterior Distributions of Consensus Answers', fontsize=16)
    plt.tight_layout()
    plt.savefig('consensus_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return consensus_df

def compare_with_majority_vote(consensus_df, data):
    """
    Compare CCT consensus with simple majority vote.
    
    Parameters:
    -----------
    consensus_df : pd.DataFrame
        DataFrame containing consensus answers
    data : numpy.ndarray
        The original data matrix
    """
    print("\n=== Comparison with Majority Vote ===")
    
    # Calculate majority vote for each question
    majority_vote = (data.mean(axis=0) > 0.5).astype(int)
    
    # Compare with CCT consensus
    consensus_answers = consensus_df['Consensus_Answer'].values
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Question': consensus_df['Question'],
        'CCT_Consensus': consensus_answers,
        'Majority_Vote': majority_vote,
        'Agreement': consensus_answers == majority_vote
    })
    
    print("\nComparison of CCT Consensus vs Majority Vote:")
    print(comparison_df.to_string(index=False))
    
    # Count agreements
    agreement_count = comparison_df['Agreement'].sum()
    total_questions = len(comparison_df)
    agreement_percentage = (agreement_count / total_questions) * 100
    
    print(f"\nAgreement: {agreement_count}/{total_questions} questions ({agreement_percentage:.1f}%)")
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    questions = range(len(consensus_df))
    width = 0.35
    
    plt.bar([q - width/2 for q in questions], consensus_answers, width, 
            label='CCT Consensus', alpha=0.8)
    plt.bar([q + width/2 for q in questions], majority_vote, width, 
            label='Majority Vote', alpha=0.8)
    
    plt.xlabel('Question')
    plt.ylabel('Answer (0 or 1)')
    plt.title('CCT Consensus vs Majority Vote')
    plt.xticks(questions, consensus_df['Question'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('consensus_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

#help of AI used to 

def main():
    """
    Main function to run the complete CCT analysis.
    """
    print("=== Cultural Consensus Theory (CCT) Analysis ===\n")
    
    # 1. Load the data
    data_path = Path('../data/plant_knowledge.csv')
    data = load_plant_knowledge_data(data_path)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # 2. Create the model
    print("\n=== Creating CCT Model ===")
    model = create_cct_model(data)
    
    # Print model structure
    print("\nModel structure:")
    print(model)
    
    # 3. Run inference
    print("\n=== Running MCMC Inference ===")
    trace = run_inference(model, draws=2000, chains=4, tune=1000)
    
    # 4. Analyze convergence
    analyze_convergence(trace)
    
    # 5. Analyze competence
    competence_df = analyze_competence(trace, data)
    
    # 6. Analyze consensus
    consensus_df = analyze_consensus(trace, data)
    
    # 7. Compare with majority vote
    comparison_df = compare_with_majority_vote(consensus_df, data)
    
    # Save results to CSV files
    competence_df.to_csv('competence_results.csv', index=False)
    consensus_df.to_csv('consensus_results.csv', index=False)
    comparison_df.to_csv('consensus_comparison.csv', index=False)
    
    print("\n=== Analysis Complete ===")
    print("Results saved to CSV files and plots saved as PNG files.")

if __name__ == "__main__":
    main()
    
    
    #overall code was put through Claude to get some more help on structure and fix small details 
    
    