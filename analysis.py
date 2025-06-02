import re
from collections import Counter
import numpy as np
import scipy.stats as stats

def parse_result_line(line):
    """Parse a result line with z-scores and detection results."""
    result = {}
    pairs = re.findall(r'([^,]+),([^,]+),', line)
    for key, value in pairs:
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result

def get_thresholds(fpr_tau, num_keys):
    """Get both uncorrected and Sidak-corrected z-score thresholds.
    
    Args:
        fpr_tau: Desired false positive rate for spoofing detection (e.g., 0.01, 0.001, 0.00003)
        num_keys: Number of keys for Sidak correction
        
    Returns:
        tuple: (uncorrected_threshold, sidak_corrected_threshold)
    """
    # Uncorrected threshold
    uncorrected_threshold = stats.norm.ppf(1 - fpr_tau)
    
    # Sidak-corrected threshold
    if num_keys <= 1:
        sidak_threshold = uncorrected_threshold
    else:
        sidak_alpha = 1 - (1 - fpr_tau)**(1/num_keys)
        sidak_threshold = stats.norm.ppf(1 - sidak_alpha)
    
    return uncorrected_threshold, sidak_threshold

def compute_secondary_threshold(num_keys=4, tau=0.5):
    """Compute theoretical secondary threshold."""
    if num_keys <= 1:
        return None
    return np.sqrt(2 * np.log(num_keys - 1)) + tau

def analyze_spoofing_success(file_path, num_keys=4, fpr_tau=0.01, 
                            secondary_threshold=None, reject=False, 
                            ethics_threshold=6.5):
    """Analyze spoofing success rates for both corrected and uncorrected thresholds.
    
    Args:
        file_path: Path to results file
        num_keys: Number of watermarking keys
        fpr_tau: Desired false positive rate for spoofing detection (e.g., 0.01)
        secondary_threshold: Secondary threshold for Algorithm 3
        reject: Whether to apply ethics-based rejection
        ethics_threshold: Ethics score threshold
        
    Returns:
        dict: Results for both corrected and uncorrected analysis
    """
    print(f"Analyzing: {file_path}")
    print(f"Keys: {num_keys}, FPR threshold: {fpr_tau}")
    
    # Get both thresholds
    uncorrected_tau, corrected_tau = get_thresholds(fpr_tau, num_keys)
    
    print(f"Uncorrected threshold: {uncorrected_tau:.3f}")
    print(f"Sidak-corrected threshold: {corrected_tau:.3f}")
    
    # Initialize counters for both analyses
    total_samples = 0
    rejected_samples = 0
    ethics_scores = []
    
    # Counters: [uncorrected, corrected]
    alg1_detected = [0, 0]  # Highest z-score
    alg2_detected = [0, 0]  # Exactly one key
    alg3_detected = [0, 0]  # Secondary threshold
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = parse_result_line(line)
            total_samples += 1
            
            # Handle ethics
            ethics_grade = data.get('GPT_Ethics')
            if ethics_grade is not None:
                ethics_scores.append(ethics_grade)
            
            is_rejected = (reject and ethics_grade is not None and 
                          ethics_grade < ethics_threshold)
            if is_rejected:
                rejected_samples += 1
            
            passes_ethics = not reject or not is_rejected
            
            # Extract z-scores
            z_scores = []
            for i in range(num_keys):
                z_key = f'Z_{i}'
                if z_key in data:
                    z_scores.append(data[z_key])
            
            if not z_scores or not passes_ethics:
                continue
            
            z_scores.sort(reverse=True)
            highest_z = z_scores[0]
            
            # Calculate detection counts for both thresholds
            uncorrected_count = sum(1 for z in z_scores if z >= uncorrected_tau)
            corrected_count = sum(1 for z in z_scores if z >= corrected_tau)
            
            # Algorithm 1: Highest z-score detection
            if highest_z >= uncorrected_tau:
                alg1_detected[0] += 1
            if highest_z >= corrected_tau:
                alg1_detected[1] += 1
            
            # Algorithm 2: Exactly one key detected
            if uncorrected_count == 1:
                alg2_detected[0] += 1
            if corrected_count == 1:
                alg2_detected[1] += 1
            
            # Algorithm 3: Secondary threshold
            if len(z_scores) > 1 and secondary_threshold is not None:
                second_highest = z_scores[1]
                if highest_z >= uncorrected_tau and second_highest < secondary_threshold:
                    alg3_detected[0] += 1
                if highest_z >= corrected_tau and second_highest < secondary_threshold:
                    alg3_detected[1] += 1
    
    # Calculate success rates
    def get_rates(counts):
        return [count / total_samples for count in counts]
    
    results = {
        'config': {
            'total_samples': total_samples,
            'rejected_samples': rejected_samples,
            'rejection_rate': rejected_samples / total_samples if total_samples > 0 else 0,
            'num_keys': num_keys,
            'fpr_tau': fpr_tau,
            'uncorrected_threshold': uncorrected_tau,
            'corrected_threshold': corrected_tau,
            'secondary_threshold': secondary_threshold
        },
        'uncorrected': {
            'algorithm1': get_rates(alg1_detected)[0],
            'algorithm2': get_rates(alg2_detected)[0],
            'algorithm3': get_rates(alg3_detected)[0]
        },
        'corrected': {
            'algorithm1': get_rates(alg1_detected)[1],
            'algorithm2': get_rates(alg2_detected)[1],
            'algorithm3': get_rates(alg3_detected)[1]
        }
    }
    
    # Add ethics info if available
    if ethics_scores:
        results['ethics'] = {
            'average': np.mean(ethics_scores),
            'min': min(ethics_scores),
            'max': max(ethics_scores),
            'below_threshold': np.mean([s < ethics_threshold for s in ethics_scores])
        }
    
    return results

def print_results(results):
    """Print analysis results in a clean format."""
    config = results['config']
    
    print(f"\n{'='*60}")
    print("SPOOFING SUCCESS RATES")
    print(f"{'='*60}")
    print(f"Total samples: {config['total_samples']}")
    print(f"Rejected (ethics): {config['rejected_samples']} ({config['rejection_rate']:.1%})")
    
    if 'ethics' in results:
        eth = results['ethics']
        print(f"Ethics avg: {eth['average']:.2f} (range: {eth['min']:.2f}-{eth['max']:.2f})")
    
    print(f"\nFPR threshold: {config['fpr_tau']}")
    print(f"Uncorrected threshold: {config['uncorrected_threshold']:.3f}")
    print(f"Sidak-corrected threshold: {config['corrected_threshold']:.3f}")
    
    # Print comparison table
    print(f"\n{'Algorithm':<25} {'Uncorrected':<12} {'Corrected':<12} {'Difference':<12}")
    print("-" * 61)
    
    algorithms = [
        ('Algorithm 1 (Highest Z)', 'algorithm1'),
        ('Algorithm 2 (Exactly One)', 'algorithm2'),
        ('Algorithm 3 (Secondary)', 'algorithm3')
    ]
    
    for name, key in algorithms:
        uncorr = results['uncorrected'][key]
        corr = results['corrected'][key]
        diff = uncorr - corr
        print(f"{name:<25} {uncorr:<12.4f} {corr:<12.4f} {diff:<12.4f}")

def analyze_file(file_path, num_keys=4, fpr_tau=0.01, tau=0.5, 
                reject=False, ethics_threshold=6.5):
    """Main analysis function.
    
    Args:
        file_path: Path to your results file
        num_keys: Number of watermarking keys used
        fpr_tau: Desired false positive rate for spoofing detection (e.g., 0.01, 0.001, 0.00003)
        tau: Parameter for secondary threshold
        reject: Apply ethics-based rejection
        ethics_threshold: Ethics score cutoff
        
    Returns:
        dict: Analysis results for both corrected and uncorrected thresholds
    """
    # Calculate secondary threshold
    secondary_threshold = (compute_secondary_threshold(num_keys, tau) 
                          if num_keys > 1 else None)
    
    # Run analysis
    results = analyze_spoofing_success(
        file_path=file_path,
        num_keys=num_keys,
        fpr_tau=fpr_tau,
        secondary_threshold=secondary_threshold,
        reject=reject,
        ethics_threshold=ethics_threshold
    )
    
    # Display results
    print_results(results)
    
    return results

if __name__ == "__main__":
    # Example usage
    analyze_file(
        file_path="/ephemeral/home/taremu/projects/watermark-supression/results/Selfhash/1_baseline/4spoofing_realharmfulq-50_8.0.txt",
        num_keys=4,
        fpr_tau=0.01,  # 1% false positive rate for spoofing detection
        reject=False
    )