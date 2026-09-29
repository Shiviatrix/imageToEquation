import os
import subprocess
from mlx_lm import load, generate

def load_framework():
    path = "/Users/babayaga/.gemini/antigravity-ide/brain/90533dfc-b6d3-4690-bd2b-b753925d4ef6/yeganeh_mathematical_framework.md"
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read framework file: {e}")
        return "You are an expert mathematical artist."

def generate_math_art(prompt, framework_text):
    print("Loading Qwen2.5-Coder-1.5B-Instruct into MLX Unified Memory...")
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct" 
    model, tokenizer = load(model_id)

    system_message = f"""You are an autonomous Agentic AI that generates photorealistic procedural mathematical art using pure PyTorch scalar fields.
You must strictly follow the mathematical framework outlined below:

--- FRAMEWORK START ---
{framework_text}
--- FRAMEWORK END ---

OUTPUT INSTRUCTIONS:
1. Generate a COMPLETE, executable python script that imports torch, numpy, and matplotlib.pyplot.
2. The script must compute a high-resolution spatial grid (e.g. 1000x1000) mapped to X, Y Cartesian and R, Theta Polar arrays.
3. The script MUST contain a function `render_art()` that computes the mathematical art. It should save the final plt image to "scratch/llm_art_output.png".
4. Ensure the math relies heavily on coordinate warping, heightmaps (Z-buffering), nested exponentials for anti-aliasing, and finite-difference lighting for photorealism.
5. Output ONLY the python code inside a ```python ``` block. No other text. End the file with a call to `render_art()`."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("Generating mathematical script via MLX... (should be extremely fast on M4!)")
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt_text, 
        max_tokens=2048,
        verbose=True
    )
    
    return response

def extract_and_run(response):
    script_filename = "scratch/llm_generated_art.py"
    
    code = response
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
        
    with open(script_filename, "w") as f:
        f.write(code)
        
    print(f"Saved generated script to {script_filename}")
    print("Executing script...")
    
    result = subprocess.run(["python3", script_filename], capture_output=True, text=True)
    if result.returncode == 0:
        print("Success! Art rendered.")
    else:
        print("Error during execution:")
        print(result.stderr)
        print("Stdout:")
        print(result.stdout)

if __name__ == "__main__":
    framework = load_framework()
    
    prompt = "Create a procedural mathematical drawing of a glowing blue star surrounded by a nebula."
    print(f"\nUser Prompt: {prompt}\n")
    
    response = generate_math_art(prompt, framework)
    extract_and_run(response)
