import os
import subprocess
import random
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import time
import json
import argparse

# Constants
ABC_PATH = "./abc"
DESIGNS_DIR = "designs"
RESULTS_DIR = "results"
MODEL_DIR = "models"
OUTPUT_LOG = "final_area_results.txt"

# ABC Commands - these are the actions our RL agent can take
ABC_COMMANDS = [
    "balance",
    "rewrite",
    "refactor",
    "rewrite -z",
    "refactor -z",
    "resub",
    "resub -z",
    "balance",
    "fraig", 
    "strash",
    "resyn",
    "resyn2",
    "balance"
]

# Create required directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Utility functions
def strip_ansi_codes(text):
    """Remove ANSI escape codes from ABC output."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def run_abc_and_get_area(design_path, recipe):
    """Run ABC tool with the provided recipe and extract area from the output."""
    abc_script = f"""
read_bench {design_path}
{recipe}
print_stats
quit
"""
    with open("temp_script.abc", "w") as f:
        f.write(abc_script)
    
    result = subprocess.run([ABC_PATH, "-f", "temp_script.abc"], 
                            capture_output=True, text=True)
    stdout = strip_ansi_codes(result.stdout)
    
    area = None
    for line in stdout.splitlines():
        if "and =" in line:
            match = re.search(r"and\s*=\s*(\d+)", line)
            if match:
                area = int(match.group(1))
                break
    
    return area

def get_raw_area(design_path):
    """Get the raw area of a design without applying any recipe."""
    abc_script = f"""
read_bench {design_path}
print_stats
quit
"""
    with open("tempp_script.abc", "w") as f:
        f.write(abc_script)
    
    result = subprocess.run([ABC_PATH, "-f", "tempp_script.abc"], 
                            capture_output=True, text=True)
    stdout = strip_ansi_codes(result.stdout)
    
    area = None
    for line in stdout.splitlines():
        if " nd =" in line:
            match = re.search(r"nd\s*=\s*(\d+)", line)
            if match:
                area = int(match.group(1))
                break
    
    return area


def get_initial_area(design_path):
    """Get the initial area of a design."""
    return get_raw_area(design_path)

def generate_random_recipe():
    """Generate a random synthesis recipe."""
    length = random.randint(1, 15)
    recipe_cmds = []
    recipe_cmds.append("strash")  # Always start with strash
    
    for _ in range(length):
        # With 8% chance, stop early
        if random.random() < 0.08:
            break
        recipe_cmds.append(random.choice(ABC_COMMANDS))
    
    return "; ".join(recipe_cmds)

# Neural Network for RL
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

class CircuitSynthesisRL:
    def __init__(self, design_dir, max_recipe_length=15):
        self.design_dir = design_dir
        self.max_recipe_length = max_recipe_length
        self.commands = ABC_COMMANDS
        self.num_commands = len(self.commands)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get all bench files
        self.bench_files = [f for f in os.listdir(design_dir) if f.endswith(".bench")]
        if not self.bench_files:
            raise ValueError(f"No .bench files found in {design_dir}")
        
        # Calculate initial areas for all designs
        self.initial_areas = {}
        for bench_file in self.bench_files:
            bench_path = os.path.join(design_dir, bench_file)
            initial_area = get_initial_area(bench_path)
            if initial_area is not None:
                self.initial_areas[bench_file] = initial_area
                print(f"Initial area for {bench_file}: {initial_area}")
            else:
                print(f"Warning: Could not get initial area for {bench_file}")
        
        # Setup RL components
        self.state_size = 3  # [current_recipe_length, last_area_reduction_ratio, design_encoding]
        self.action_size = self.num_commands
        self.hidden_size = 128
        self.policy_net = PolicyNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.gamma = 0.99  # Discount factor
        
        # Design encoding - one-hot encoding of different designs
        self.design_encodings = {bench_file: i for i, bench_file in enumerate(self.bench_files)}
        
    def get_state(self, design_file, recipe_length, last_area_ratio):
        """Create a state representation."""
        # Normalize recipe length
        norm_recipe_length = recipe_length / self.max_recipe_length
        
        # One-hot encode the design
        design_encoding = self.design_encodings[design_file] / len(self.bench_files)
        
        return torch.tensor([[norm_recipe_length, last_area_ratio, design_encoding]], 
                          dtype=torch.float32, device=self.device)
    
    def select_action(self, state):
        """Select an action based on the current policy."""
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def calculate_reward(self, old_area, new_area):
        """Calculate reward based on area reduction."""
        if new_area is None:
            return -1.0  # Penalty for failed action
        
        # Area reduction ratio
        area_reduction = (old_area - new_area) / old_area
        
        # Reward is the area reduction percentage
        return area_reduction * 10  # Scale for better training signals
    
    def train_episode(self, design_file):
        """Train the agent for one episode on a specific design."""
        self.policy_net.train()  # Set to training mode
        design_path = os.path.join(self.design_dir, design_file)
        initial_area = self.initial_areas[design_file]
        
        # Episode storage
        saved_log_probs = []
        rewards = []
        
        # Start with strash
        recipe = ["strash"]
        current_area = initial_area
        
        # Execute steps until we reach max length or early stopping
        for step in range(self.max_recipe_length - 1):  # -1 because we already used strash
            # Get current state
            area_ratio = 1.0 if initial_area == current_area else current_area / initial_area
            state = self.get_state(design_file, len(recipe), area_ratio)
            
            # Select action
            action_idx, log_prob = self.select_action(state)
            saved_log_probs.append(log_prob)
            action = self.commands[action_idx]
            
            # Execute action
            new_recipe = "; ".join(recipe + [action])
            new_area = run_abc_and_get_area(design_path, new_recipe)
            
            # Calculate reward
            reward = self.calculate_reward(current_area, new_area)
            rewards.append(reward)
            
            # Update state
            if new_area is not None:
                recipe.append(action)
                current_area = new_area
            
            # Early stopping if no improvement for a while
            if len(rewards) >= 3 and all(r <= 0 for r in rewards[-3:]):
                break
        
        # Ensure we have at least one step
        if not rewards:
            recipe_str = "; ".join(recipe)
            return recipe_str, current_area, initial_area
        
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Convert to tensor and normalize
        returns = torch.tensor(returns, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Check if we have valid log_probs
        if policy_loss:
            policy_loss = torch.cat(policy_loss).sum()
            
            # Optimize the model
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
        
        recipe_str = "; ".join(recipe)
        return recipe_str, current_area, initial_area
    
    def train(self, num_episodes=1000):
        """Train the agent for multiple episodes."""
        results = []
        best_recipes = {}
        
        for episode in range(num_episodes):
            # Select a random design file
            design_file = random.choice(self.bench_files)
            
            # Train for one episode
            recipe, final_area, initial_area = self.train_episode(design_file)
            
            # Calculate improvement
            improvement = (initial_area - final_area) / initial_area * 100
            
            # Log results
            result = {
                "episode": episode + 1,
                "design": design_file,
                "initial_area": initial_area,
                "final_area": final_area,
                "improvement": improvement,
                "recipe": recipe
            }
            results.append(result)
            
            # Update best recipe for this design
            if design_file not in best_recipes or final_area < best_recipes[design_file]["final_area"]:
                best_recipes[design_file] = {
                    "recipe": recipe,
                    "final_area": final_area,
                    "improvement": improvement
                }
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} - Design: {design_file}")
                print(f"  Initial area: {initial_area}, Final area: {final_area}")
                print(f"  Improvement: {improvement:.2f}%")
                print(f"  Recipe: {recipe}")
                print("-" * 50)
            
            # Save model checkpoint
            if (episode + 1) % 100 == 0:
                self.save_model(f"{MODEL_DIR}/model_episode_{episode+1}.pth")
                self.save_results(results, f"{RESULTS_DIR}/results_episode_{episode+1}.json")
        
        # Save final model and results
        self.save_model(f"{MODEL_DIR}/final_model.pth")
        self.save_results(results, f"{RESULTS_DIR}/final_results.json")
        self.save_best_recipes(best_recipes, f"{RESULTS_DIR}/best_recipes.json")
        
        return best_recipes
    
    def save_model(self, path):
        """Save the trained model."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model."""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
        print(f"Model loaded from {path}")
    
    def save_results(self, results, path):
        """Save training results."""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")
    
    def save_best_recipes(self, best_recipes, path):
        """Save the best recipes for each design."""
        with open(path, 'w') as f:
            json.dump(best_recipes, f, indent=2)
        print(f"Best recipes saved to {path}")
    
    def get_optimal_recipe(self, design_file):
        """Infer the optimal recipe for a given design using the trained model."""
        self.policy_net.eval()  # Set to evaluation mode
        design_path = os.path.join(self.design_dir, design_file)
        initial_area = get_initial_area(design_path)
        
        if initial_area is None:
            print(f"Error: Could not get initial area for {design_file}")
            return None, None, None
        
        # Start with strash
        recipe = ["strash"]
        current_area = initial_area
        
        # Execute steps until we reach max length or early stopping
        for step in range(self.max_recipe_length - 1):  # -1 because we already used strash
            # Get current state
            area_ratio = 1.0 if initial_area == current_area else current_area / initial_area
            state = self.get_state(design_file, len(recipe), area_ratio)
            
            # Select best action
            with torch.no_grad():
                probs = self.policy_net(state)
                action_idx = torch.argmax(probs).item()
            
            action = self.commands[action_idx]
            
            # Execute action
            new_recipe = "; ".join(recipe + [action])
            new_area = run_abc_and_get_area(design_path, new_recipe)
            
            # Update state
            if new_area is not None and new_area < current_area:
                recipe.append(action)
                current_area = new_area
            else:
                # Stop if no improvement
                break
        
        final_recipe = "; ".join(recipe)
        return final_recipe, initial_area, current_area

def main():
    parser = argparse.ArgumentParser(description='Circuit Synthesis Optimization with RL')
    parser.add_argument('--mode', choices=['train', 'infer'], default='train', help='Operation mode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--model', type=str, default=None, help='Path to model file for inference')
    parser.add_argument('--design', type=str, default=None, help='Design file for inference')
    args = parser.parse_args()
    
    # Set manual seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    rl_agent = CircuitSynthesisRL(DESIGNS_DIR)
    
    if args.mode == 'train':
        print(f"Starting training for {args.episodes} episodes...")
        best_recipes = rl_agent.train(num_episodes=args.episodes)
        
        # Print summary of best recipes
        print("\n===== Best Synthesis Recipes =====")
        for design, data in best_recipes.items():
            print(f"Design: {design}")
            print(f"  Initial area: {rl_agent.initial_areas[design]}")
            print(f"  Final area: {data['final_area']}")
            print(f"  Improvement: {data['improvement']:.2f}%")
            print(f"  Recipe: {data['recipe']}")
            print("-" * 50)
    
    elif args.mode == 'infer':
        if args.model is None:
            model_path = f"{MODEL_DIR}/model_episode_300.pth"  #final_model.pth
        else:
            model_path = args.model
        
        # Load the trained model
        rl_agent.load_model(model_path)
        
        if args.design:
            # Infer for a specific design
            design_file = args.design
            recipe, initial_area, final_area = rl_agent.get_optimal_recipe(design_file)
            
            if recipe:
                improvement = (initial_area - final_area) / initial_area * 100
                print(f"\n===== Optimal Synthesis Recipe for {design_file} =====")
                print(f"Initial area: {initial_area}")
                print(f"Final area: {final_area}")
                print(f"Improvement: {improvement:.2f}%")
                print(f"Recipe: {recipe}")
            else:
                print(f"Could not generate a recipe for {design_file}")
        else:
            # Infer for all designs
            print("\n===== Optimal Synthesis Recipes =====")
            for design_file in rl_agent.bench_files:
                recipe, initial_area, final_area = rl_agent.get_optimal_recipe(design_file)
                
                if recipe:
                    improvement = (initial_area - final_area) / initial_area * 100
                    print(f"Design: {design_file}")
                    print(f"  Initial area: {initial_area}")
                    print(f"  Final area: {final_area}")
                    print(f"  Improvement: {improvement:.2f}%")
                    print(f"  Recipe: {recipe}")
                    print("-" * 50)

if __name__ == "__main__":
    main()
