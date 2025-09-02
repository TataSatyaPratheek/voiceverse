import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from .data_utils import TeluguEnglishDataset, create_persona_calibration_dataset
from .persona import PersonaVectorManager, PersonaConfig
from .tts import StyleTTSWrapper
from .vocoder import HiFiGANVocoder

class PersonaFineTuner:
    """Fine-tune persona vectors on Telugu/English data"""
    
    def __init__(self, tts_model: StyleTTSWrapper, vocoder: HiFiGANVocoder,
                 persona_manager: PersonaVectorManager):
        self.tts_model = tts_model
        self.vocoder = vocoder
        self.persona_manager = persona_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def finetune_persona(self, persona_name: str, dataset: TeluguEnglishDataset,
                        epochs: int = 50, batch_size: int = 4, 
                        learning_rate: float = 0.001) -> Dict[str, float]:
        """Fine-tune persona vector on target dataset"""
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"persona_finetune_{persona_name}"):
            mlflow.log_params({
                "persona_name": persona_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            })
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Get persona vector and make it trainable
            persona_vector = self.persona_manager.personas[persona_name].clone()
            persona_vector.requires_grad_(True)
            
            # Optimizer for persona vector only
            optimizer = optim.Adam([persona_vector], lr=learning_rate)
            
            losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                    optimizer.zero_grad()
                    
                    # Forward pass through TTS
                    generated_mel = self._generate_batch(batch, persona_vector)
                    
                    # Compute loss against target mel
                    loss = self._compute_loss(generated_mel, batch['mel'])
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                losses.append(avg_loss)
                
                # Log metrics
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                
                # Validate every 10 epochs
                if (epoch + 1) % 10 == 0:
                    val_metrics = self._validate_persona(persona_vector, dataset)
                    for metric, value in val_metrics.items():
                        mlflow.log_metric(f"val_{metric}", value, step=epoch)
                
                print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
            
            # Update persona vector
            self.persona_manager.personas[persona_name] = persona_vector.detach()
            
            # Save final model
            persona_path = f"artifacts/persona_{persona_name}_finetuned.pt"
            torch.save(persona_vector.detach(), persona_path)
            mlflow.log_artifact(persona_path)
            
            return {"final_loss": losses[-1], "total_epochs": epochs}
    
    def _generate_batch(self, batch: Dict[str, torch.Tensor], 
                       persona_vector: torch.Tensor) -> torch.Tensor:
        """Generate mel spectrograms for batch using persona vector"""
        batch_mels = []
        
        for i in range(len(batch['text'])):
            text = batch['text'][i]
            
            # Create style controls from persona config
            persona_name = list(self.persona_manager.personas.keys())[0]  # Get any persona name
            config = self.persona_manager.configs[persona_name]
            style_controls = {
                "tempo": config.tempo,
                "energy": config.energy,
                "warmth": config.warmth,
                "formality": config.formality
            }
            
            # Generate mel using current persona vector
            mel = self.tts_model.synthesize(text, persona_vector, style_controls)
            batch_mels.append(torch.FloatTensor(mel))
        
        return torch.stack(batch_mels)
    
    def _compute_loss(self, generated_mel: torch.Tensor, 
                     target_mel: torch.Tensor) -> torch.Tensor:
        """Compute loss between generated and target mel spectrograms"""
        # L1 + L2 loss combination
        l1_loss = nn.L1Loss()(generated_mel, target_mel)
        l2_loss = nn.MSELoss()(generated_mel, target_mel)
        
        return l1_loss + 0.1 * l2_loss
    
    def _validate_persona(self, persona_vector: torch.Tensor, 
                         dataset: TeluguEnglishDataset) -> Dict[str, float]:
        """Validate persona vector on held-out data"""
        # Implement validation metrics: MOS prediction, style consistency, etc.
        return {
            "style_consistency": 0.85,  # placeholder
            "pronunciation_accuracy": 0.90,  # placeholder
            "naturalness_score": 0.88  # placeholder
        }

class AdaptivePersonaTrainer:
    """Advanced trainer with adaptive persona vector updates"""
    
    def __init__(self, base_models: Dict[str, nn.Module]):
        self.models = base_models
        self.persona_history = []
        
    def adaptive_train(self, target_characteristics: Dict[str, float],
                      training_data: DataLoader, validation_data: DataLoader,
                      max_iterations: int = 1000) -> torch.Tensor:
        """Adaptively train persona vector to match target characteristics"""
        
        # Initialize persona vector
        persona_vector = torch.randn(256, requires_grad=True)
        optimizer = optim.Adam([persona_vector], lr=0.001)
        
        best_persona = None
        best_score = float('-inf')
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Compute multi-objective loss
            loss_components = self._compute_multi_objective_loss(
                persona_vector, target_characteristics, training_data)
            
            total_loss = sum(loss_components.values())
            total_loss.backward()
            optimizer.step()
            
            # Validation
            if iteration % 50 == 0:
                val_score = self._evaluate_persona(persona_vector, validation_data)
                
                if val_score > best_score:
                    best_score = val_score
                    best_persona = persona_vector.clone().detach()
                
                print(f"Iteration {iteration}: Loss = {total_loss.item():.6f}, "
                      f"Val Score = {val_score:.4f}")
        
        return best_persona if best_persona is not None else persona_vector.detach()
    
    def _compute_multi_objective_loss(self, persona_vector: torch.Tensor,
                                    targets: Dict[str, float],
                                    data: DataLoader) -> Dict[str, torch.Tensor]:
        """Compute losses for multiple style objectives"""
        return {
            "style_loss": torch.tensor(0.1),  # placeholder
            "quality_loss": torch.tensor(0.05),  # placeholder
            "consistency_loss": torch.tensor(0.02)  # placeholder
        }
    
    def _evaluate_persona(self, persona_vector: torch.Tensor,
                         validation_data: DataLoader) -> float:
        """Evaluate persona vector on validation set"""
        return np.random.random()  # placeholder score
