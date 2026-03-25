import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

class RealisticSepsisDatasetGenerator:
    """
    Generate realistic synthetic sepsis monitoring dataset for preterm neonates
    Based on actual clinical vital sign ranges and biomarkers
    
    Features (10 clinical parameters):
    1. Heart Rate (bpm): 120-180 normal
    2. Respiratory Rate (breaths/min): 40-60 normal
    3. Temperature (°C): 36.5-37.4 normal
    4. Systolic Blood Pressure (mmHg): 40-70 normal for preterm
    5. O2 Saturation (%): 90-95% normal
    6. White Blood Cell Count (×10³/μL): 9-30 normal
    7. Immature-to-Total Neutrophil Ratio: <0.2 normal
    8. C-Reactive Protein (mg/L): <10 normal
    9. Procalcitonin (ng/mL): <0.5 normal
    10. Platelet Count (×10³/μL): 150-450 normal
    """
    
    def __init__(self, n_samples=1000, sequence_length=100, normal_ratio=0.7, random_state=42):
        np.random.seed(random_state)
        self.n_samples = n_samples
        self.sequence_length = sequence_length  # 100 time points (e.g., hourly monitoring)
        self.normal_ratio = normal_ratio
        self.n_normal = int(n_samples * normal_ratio)
        self.n_sepsis = n_samples - self.n_normal
        
        self.feature_names = [
            'Heart_Rate', 'Respiratory_Rate', 'Temperature', 
            'Systolic_BP', 'O2_Saturation', 'WBC_Count',
            'IT_Ratio', 'CRP', 'Procalcitonin', 'Platelet_Count'
        ]
        
    def generate_normal_vital_signs(self):
        """
        Generate normal vital signs for healthy preterm neonates
        Stable patterns with minimal variation
        """
        sequence = np.zeros((self.n_normal, self.sequence_length, 10))
        
        for i in range(self.n_normal):
            # 1. Heart Rate: 120-180 bpm, stable
            baseline_hr = np.random.normal(140, 8)
            sequence[i, :, 0] = baseline_hr + np.random.normal(0, 3, self.sequence_length)
            sequence[i, :, 0] = np.clip(sequence[i, :, 0], 120, 180)
            
            # 2. Respiratory Rate: 40-60 breaths/min, stable
            baseline_rr = np.random.normal(50, 4)
            sequence[i, :, 1] = baseline_rr + np.random.normal(0, 2, self.sequence_length)
            sequence[i, :, 1] = np.clip(sequence[i, :, 1], 40, 60)
            
            # 3. Temperature: 36.5-37.4°C, stable
            baseline_temp = np.random.normal(37.0, 0.2)
            sequence[i, :, 2] = baseline_temp + np.random.normal(0, 0.1, self.sequence_length)
            sequence[i, :, 2] = np.clip(sequence[i, :, 2], 36.5, 37.4)
            
            # 4. Systolic BP: 40-70 mmHg (preterm range), stable
            baseline_sbp = np.random.normal(55, 5)
            sequence[i, :, 3] = baseline_sbp + np.random.normal(0, 2, self.sequence_length)
            sequence[i, :, 3] = np.clip(sequence[i, :, 3], 40, 70)
            
            # 5. O2 Saturation: 90-95%, stable
            baseline_o2 = np.random.normal(93, 1.5)
            sequence[i, :, 4] = baseline_o2 + np.random.normal(0, 0.5, self.sequence_length)
            sequence[i, :, 4] = np.clip(sequence[i, :, 4], 90, 95)
            
            # 6. WBC Count: 9-30 × 10³/μL, stable
            baseline_wbc = np.random.normal(15, 4)
            sequence[i, :, 5] = baseline_wbc + np.random.normal(0, 1.5, self.sequence_length)
            sequence[i, :, 5] = np.clip(sequence[i, :, 5], 9, 30)
            
            # 7. I:T Ratio: <0.2, normal low
            baseline_it = np.random.normal(0.08, 0.04)
            sequence[i, :, 6] = baseline_it + np.random.normal(0, 0.02, self.sequence_length)
            sequence[i, :, 6] = np.clip(sequence[i, :, 6], 0.01, 0.2)
            
            # 8. CRP: <10 mg/L, stable
            baseline_crp = np.random.normal(3, 2)
            sequence[i, :, 7] = baseline_crp + np.random.normal(0, 0.5, self.sequence_length)
            sequence[i, :, 7] = np.clip(sequence[i, :, 7], 0.1, 10)
            
            # 9. Procalcitonin: <0.5 ng/mL, stable
            baseline_pct = np.random.normal(0.1, 0.05)
            sequence[i, :, 8] = baseline_pct + np.random.normal(0, 0.02, self.sequence_length)
            sequence[i, :, 8] = np.clip(sequence[i, :, 8], 0.01, 0.5)
            
            # 10. Platelet Count: 150-450 × 10³/μL, stable
            baseline_plt = np.random.normal(250, 50)
            sequence[i, :, 9] = baseline_plt + np.random.normal(0, 15, self.sequence_length)
            sequence[i, :, 9] = np.clip(sequence[i, :, 9], 150, 450)
        
        return sequence
    
    def generate_sepsis_vital_signs(self):
        """
        Generate abnormal vital signs indicative of late-onset sepsis
        Progressive deterioration pattern over 100 time points
        """
        sequence = np.zeros((self.n_sepsis, self.sequence_length, 10))
        
        for i in range(self.n_sepsis):
            # Progressive deterioration pattern (0 to 1)
            for t in range(self.sequence_length):
                # Progress factor: 0 (healthy) -> 1 (severe sepsis)
                progress = t / self.sequence_length
                
                # Sepsis indicator: gradual deterioration
                deterioration_factor = np.sin(progress * np.pi / 2)  # Smooth curve 0->1
                
                # 1. Heart Rate increases (tachycardia) - early sign of sepsis
                normal_hr = np.random.normal(140, 8)
                sepsis_hr = normal_hr + (40 * deterioration_factor)
                sequence[i, t, 0] = sepsis_hr + np.random.normal(0, 3)
                sequence[i, t, 0] = np.clip(sequence[i, t, 0], 120, 200)
                
                # 2. Respiratory Rate increases (tachypnea)
                normal_rr = np.random.normal(50, 4)
                sepsis_rr = normal_rr + (25 * deterioration_factor)
                sequence[i, t, 1] = sepsis_rr + np.random.normal(0, 2)
                sequence[i, t, 1] = np.clip(sequence[i, t, 1], 40, 90)
                
                # 3. Temperature becomes unstable (fever or hypothermia - concerning)
                normal_temp = np.random.normal(37.0, 0.2)
                if np.random.rand() > 0.5:
                    # Fever pattern
                    sepsis_temp = normal_temp + (1.2 * deterioration_factor)
                else:
                    # Hypothermia pattern (more concerning in neonates)
                    sepsis_temp = normal_temp - (1.0 * deterioration_factor)
                sequence[i, t, 2] = sepsis_temp + np.random.normal(0, 0.15)
                sequence[i, t, 2] = np.clip(sequence[i, t, 2], 35.0, 39.0)
                
                # 4. Blood Pressure may decrease (hypotension - late sign)
                normal_sbp = np.random.normal(55, 5)
                sepsis_sbp = normal_sbp - (20 * deterioration_factor)
                sequence[i, t, 3] = sepsis_sbp + np.random.normal(0, 2)
                sequence[i, t, 3] = np.clip(sequence[i, t, 3], 35, 70)
                
                # 5. O2 Saturation decreases (hypoxemia)
                normal_o2 = np.random.normal(93, 1.5)
                sepsis_o2 = normal_o2 - (8 * deterioration_factor)
                sequence[i, t, 4] = sepsis_o2 + np.random.normal(0, 0.8)
                sequence[i, t, 4] = np.clip(sequence[i, t, 4], 80, 95)
                
                # 6. WBC becomes elevated (early response) or LOW (severe sepsis)
                if progress < 0.5:
                    # Early: elevated WBC
                    normal_wbc = np.random.normal(15, 4)
                    sepsis_wbc = normal_wbc + (15 * deterioration_factor * 2)
                else:
                    # Late: can drop significantly in severe sepsis
                    normal_wbc = np.random.normal(15, 4)
                    sepsis_wbc = normal_wbc + (8 * (1 - deterioration_factor))
                sequence[i, t, 5] = sepsis_wbc + np.random.normal(0, 2)
                sequence[i, t, 5] = np.clip(sequence[i, t, 5], 5, 40)
                
                # 7. I:T Ratio increases significantly (left shift - immature cells)
                normal_it = np.random.normal(0.08, 0.04)
                sepsis_it = normal_it + (0.25 * deterioration_factor)
                sequence[i, t, 6] = sepsis_it + np.random.normal(0, 0.03)
                sequence[i, t, 6] = np.clip(sequence[i, t, 6], 0.01, 0.5)
                
                # 8. CRP increases significantly (inflammatory marker)
                normal_crp = np.random.normal(3, 2)
                sepsis_crp = normal_crp + (40 * deterioration_factor)
                sequence[i, t, 7] = sepsis_crp + np.random.normal(0, 2)
                sequence[i, t, 7] = np.clip(sequence[i, t, 7], 0.1, 100)
                
                # 9. Procalcitonin increases significantly (most specific for bacterial sepsis)
                normal_pct = np.random.normal(0.1, 0.05)
                sepsis_pct = normal_pct + (2.0 * deterioration_factor)
                sequence[i, t, 8] = sepsis_pct + np.random.normal(0, 0.1)
                sequence[i, t, 8] = np.clip(sequence[i, t, 8], 0.01, 5.0)
                
                # 10. Platelet Count decreases (thrombocytopenia in severe sepsis)
                normal_plt = np.random.normal(250, 50)
                sepsis_plt = normal_plt - (150 * deterioration_factor)
                sequence[i, t, 9] = sepsis_plt + np.random.normal(0, 20)
                sequence[i, t, 9] = np.clip(sequence[i, t, 9], 50, 450)
        
        return sequence
    
    def generate_labels(self):
        """Generate binary labels (0: normal, 1: sepsis)"""
        labels = np.concatenate([
            np.zeros(self.n_normal),
            np.ones(self.n_sepsis)
        ])
        return labels
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print("Generating normal vital signs...")
        X_normal = self.generate_normal_vital_signs()
        
        print("Generating sepsis vital signs...")
        X_sepsis = self.generate_sepsis_vital_signs()
        
        # Concatenate
        X = np.concatenate([X_normal, X_sepsis], axis=0)
        y = self.generate_labels()
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def save_dataset(self, X, y, output_dir='data'):
        """Save dataset to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nSaving dataset...")
        
        # Save as 3D numpy array
        np.save(f'{output_dir}/X_data.npy', X)
        np.save(f'{output_dir}/y_labels.npy', y)
        
        # Also save flattened version for reference
        X_flat = X.reshape(X.shape[0], -1)
        feature_columns = []
        for t in range(self.sequence_length):
            for feat_name in self.feature_names:
                feature_columns.append(f"{feat_name}_T{t}")
        
        df = pd.DataFrame(X_flat, columns=feature_columns)
        df['Label'] = y
        df['Label_Name'] = df['Label'].map({0: 'Normal', 1: 'Sepsis'})
        df.to_csv(f'{output_dir}/dataset_flattened.csv', index=False)
        
        # Save metadata
        metadata = {
            'Total Samples': X.shape[0],
            'Normal Samples': int(np.sum(y == 0)),
            'Sepsis Samples': int(np.sum(y == 1)),
            'Sequence Length': X.shape[1],
            'Number of Features': X.shape[2],
            'Feature Names': self.feature_names,
            'Generated': datetime.now().isoformat()
        }
        
        metadata_df = pd.DataFrame([metadata]).T
        metadata_df.to_csv(f'{output_dir}/metadata.csv', header=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print("DATASET GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset saved to: {output_dir}/")
        print(f"\nDataset Shape: {X.shape}")
        print(f"  - Samples: {X.shape[0]}")
        print(f"  - Time Steps: {X.shape[1]}")
        print(f"  - Features per timestep: {X.shape[2]}")
        print(f"\nClass Distribution:")
        print(f"  - Normal (Class 0): {np.sum(y == 0)} ({100*np.sum(y == 0)/len(y):.1f}%)")
        print(f"  - Sepsis (Class 1): {np.sum(y == 1)} ({100*np.sum(y == 1)/len(y):.1f}%)")
        print(f"\nFeatures ({len(self.feature_names)}):")
        for i, name in enumerate(self.feature_names):
            print(f"  {i+1}. {name}")
        print(f"\nFiles Generated:")
        print(f"  - {output_dir}/X_data.npy (3D array, optimized for ML)")
        print(f"  - {output_dir}/y_labels.npy (1D labels array)")
        print(f"  - {output_dir}/dataset_flattened.csv (human-readable format)")
        print(f"  - {output_dir}/metadata.csv (dataset metadata)")
        print(f"{'='*60}\n")
        
        return metadata


def load_dataset(data_dir='data'):
    """Load previously generated dataset"""
    X = np.load(f'{data_dir}/X_data.npy')
    y = np.load(f'{data_dir}/y_labels.npy')
    return X, y


if __name__ == "__main__":
    # Generate dataset
    print("Initializing dataset generator...")
    generator = RealisticSepsisDatasetGenerator(
        n_samples=1000, 
        sequence_length=100, 
        normal_ratio=0.7,
        random_state=42
    )
    
    # Generate
    X, y = generator.generate_dataset()
    
    # Save
    generator.save_dataset(X, y, output_dir='data')
    
    # Display sample statistics
    print("\nSample Statistics:")
    print(f"X_normal mean: {X[y==0].mean(axis=(0, 1))}")
    print(f"X_sepsis mean: {X[y==1].mean(axis=(0, 1))}")
    
    # Load and verify
    print("\nVerifying saved dataset...")
    X_loaded, y_loaded = load_dataset('data')
    print(f"Loaded shape: {X_loaded.shape}")
    print("Dataset ready for preprocessing and model training!")