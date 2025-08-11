import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class QwenAttentionVisualizer:
    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat"):
        """
        Initialize the Qwen attention visualizer.
        
        Args:
            model_name: HuggingFace model identifier for Qwen model
        """
        print(f"Loading {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            pad_token='<|endoftext|>'
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True
        )
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
    def find_token_indices(self, tokens: List[str], target_phrase: str) -> List[int]:
        """
        Find token indices that correspond to a target phrase.
        
        Args:
            tokens: List of tokens from tokenizer
            target_phrase: Target phrase to find (e.g., "red cat")
            
        Returns:
            List of token indices that match the target phrase
        """
        target_tokens = self.tokenizer.tokenize(target_phrase.lower())
        target_indices = []
        
        # Convert tokens to lowercase for matching
        tokens_lower = [token.lower() for token in tokens]
        
        # Find exact matches and partial matches
        for i, token in enumerate(tokens_lower):
            # Direct token match
            if any(target_token in token for target_token in target_tokens):
                target_indices.append(i)
            # Check if token is part of target phrase
            elif any(token in target_token for target_token in target_tokens):
                target_indices.append(i)
        
        # Also try to find sequential matches
        for i in range(len(tokens_lower) - len(target_tokens) + 1):
            token_sequence = ''.join(tokens_lower[i:i+len(target_tokens)])
            target_sequence = ''.join(target_tokens)
            if target_sequence in token_sequence:
                target_indices.extend(range(i, i+len(target_tokens)))
        
        return sorted(list(set(target_indices)))
    
    def get_attention_maps(self, prompt: str) -> Tuple[torch.Tensor, List[str], Dict]:
        """
        Extract attention maps from the model for a given prompt.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Tuple of (attention_weights, tokens, model_info)
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        
        # Get tokens for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
            attentions = outputs.attentions  # Tuple of attention weights for each layer
        
        # Convert to tensor: [layers, batch, heads, seq_len, seq_len]
        attention_tensor = torch.stack(attentions)
        
        model_info = {
            'num_layers': len(attentions),
            'num_heads': attentions[0].shape[2],
            'seq_length': attentions[0].shape[3]
        }
        
        return attention_tensor, tokens, model_info
    
    def visualize_token_attention(self, 
                                prompt: str, 
                                target_phrase: str,
                                layers_to_show: Optional[List[int]] = None,
                                heads_to_show: Optional[List[int]] = None,
                                save_path: Optional[str] = None):
        """
        Visualize attention maps for specific tokens related to target phrase.
        
        Args:
            prompt: Input prompt
            target_phrase: Target phrase to analyze (e.g., "red cat")
            layers_to_show: List of layer indices to visualize (default: last 4 layers)
            heads_to_show: List of head indices to visualize (default: first 8 heads)
            save_path: Path to save the visualization
        """
        print(f"Analyzing attention for phrase: '{target_phrase}'")
        
        # Get attention maps
        attention_tensor, tokens, model_info = self.get_attention_maps(prompt)
        
        # Find target token indices
        target_indices = self.find_token_indices(tokens, target_phrase)
        
        if not target_indices:
            print(f"Warning: Could not find tokens for '{target_phrase}' in the prompt")
            print(f"Available tokens: {tokens}")
            return
        
        print(f"Found target tokens at indices: {target_indices}")
        print(f"Target tokens: {[tokens[i] for i in target_indices]}")
        
        # Set default layers and heads to show
        if layers_to_show is None:
            layers_to_show = list(range(max(0, model_info['num_layers']-4), model_info['num_layers']))
        if heads_to_show is None:
            heads_to_show = list(range(min(8, model_info['num_heads'])))
        
        # Create visualization
        fig, axes = plt.subplots(len(layers_to_show), len(heads_to_show), 
                               figsize=(4*len(heads_to_show), 4*len(layers_to_show)))
        
        if len(layers_to_show) == 1:
            axes = axes.reshape(1, -1)
        if len(heads_to_show) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, layer_idx in enumerate(layers_to_show):
            for j, head_idx in enumerate(heads_to_show):
                # Get attention weights for this layer and head
                attention_weights = attention_tensor[layer_idx, 0, head_idx].cpu().numpy()
                
                # Average attention from target tokens to all tokens
                target_attention = np.mean(attention_weights[target_indices, :], axis=0)
                
                # Create heatmap
                ax = axes[i, j] if len(layers_to_show) > 1 and len(heads_to_show) > 1 else \
                     axes[i] if len(heads_to_show) == 1 else \
                     axes[j] if len(layers_to_show) == 1 else axes
                
                # Prepare data for heatmap
                attention_matrix = attention_weights[target_indices, :]
                
                sns.heatmap(attention_matrix, 
                           xticklabels=tokens, 
                           yticklabels=[tokens[idx] for idx in target_indices],
                           cmap='Blues', 
                           ax=ax,
                           cbar=True,
                           square=False)
                
                ax.set_title(f'Layer {layer_idx}, Head {head_idx}')
                ax.set_xlabel('Tokens (Attended To)')
                ax.set_ylabel('Target Tokens (Attending From)')
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return attention_tensor, tokens, target_indices
    
    def create_attention_summary(self, 
                               prompt: str, 
                               target_phrase: str,
                               top_k: int = 10) -> Dict:
        """
        Create a summary of attention patterns for target tokens.
        
        Args:
            prompt: Input prompt
            target_phrase: Target phrase to analyze
            top_k: Number of top attended tokens to report
            
        Returns:
            Dictionary with attention analysis summary
        """
        attention_tensor, tokens, model_info = self.get_attention_maps(prompt)
        target_indices = self.find_token_indices(tokens, target_phrase)
        
        if not target_indices:
            return {"error": f"Could not find tokens for '{target_phrase}'"}
        
        summary = {
            "target_phrase": target_phrase,
            "target_tokens": [tokens[i] for i in target_indices],
            "target_indices": target_indices,
            "model_info": model_info,
            "layer_summaries": []
        }
        
        for layer_idx in range(model_info['num_layers']):
            layer_attention = attention_tensor[layer_idx, 0].cpu().numpy()  # [heads, seq, seq]
            
            # Average attention across all target tokens and heads
            target_attention = np.mean(layer_attention[:, target_indices, :], axis=(0, 1))
            
            # Find top attended tokens
            top_indices = np.argsort(target_attention)[-top_k:][::-1]
            top_tokens = [(tokens[idx], target_attention[idx], idx) for idx in top_indices]
            
            layer_summary = {
                "layer": layer_idx,
                "avg_attention_strength": float(np.mean(target_attention)),
                "max_attention": float(np.max(target_attention)),
                "top_attended_tokens": top_tokens
            }
            summary["layer_summaries"].append(layer_summary)
        
        return summary
    
    def print_attention_summary(self, summary: Dict):
        """Print a formatted attention summary."""
        if "error" in summary:
            print(summary["error"])
            return
        
        print(f"\n=== ATTENTION ANALYSIS FOR '{summary['target_phrase']}' ===")
        print(f"Target tokens: {summary['target_tokens']}")
        print(f"Model: {summary['model_info']['num_layers']} layers, {summary['model_info']['num_heads']} heads per layer")
        print("-" * 60)
        
        for layer_summary in summary['layer_summaries']:
            print(f"\nLayer {layer_summary['layer']}:")
            print(f"  Avg attention strength: {layer_summary['avg_attention_strength']:.4f}")
            print(f"  Max attention: {layer_summary['max_attention']:.4f}")
            print(f"  Top attended tokens:")
            for token, attention, idx in layer_summary['top_attended_tokens'][:5]:
                print(f"    {token:15} -> {attention:.4f} (pos {idx})")


# Example usage
def main():
    # Initialize visualizer
    # Note: Change model name based on available Qwen models
    # Examples: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat", "Qwen/Qwen-1_8B-Chat"
    visualizer = QwenAttentionVisualizer("Qwen/Qwen-1_8B-Chat")  # Using smaller model for demo
    
    # Example prompt
    prompt = "Where is the red cat on the image?"
    target_phrase = "red cat"
    
    # Get attention summary
    summary = visualizer.create_attention_summary(prompt, target_phrase, top_k=10)
    visualizer.print_attention_summary(summary)
    
    # Visualize attention maps
    # Show last 3 layers and first 4 heads
    visualizer.visualize_token_attention(
        prompt=prompt,
        target_phrase=target_phrase,
        layers_to_show=[-3, -2, -1],  # Last 3 layers
        heads_to_show=[0, 1, 2, 3],   # First 4 heads
        save_path="qwen_attention_red_cat.png"
    )


if __name__ == "__main__":
    main()
