from __future__ import annotations

import json
from pathlib import Path

from code_executor import CodeExecutor
from llm_client import LLMClient

def load_config(path: Path) -> dict:
    """Loads a JSON configuration file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    # 1. Setup paths based on your project structure
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "agent_config.json"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    print("Loading configuration...")
    config = load_config(config_path)

    print(f"Initializing LLM Client with model: {config['llm']['model']}...")
    
    # 2. Instantiate the client using config instead of .env
    llm = LLMClient(
        provider=config["llm"]["provider"], 
        model=config["llm"]["model"]
    ) 

    # 3. Instantiate the Code Executor
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=config["execution"]["timeout_seconds"],
    )

    print("Generating code... (this might take a few seconds)")

    # 4. Define our test prompts
    system_prompt = (
        "You are an expert Python developer. Write clean, working code. "
        "Do NOT output markdown formatting like ```python. Just output the raw code. and nothing alese"
    )
    user_prompt = "Write a simple Python script that prints 'Hello from the autonomous agent!' and defines a function to calculate the factorial of 5, then prints the result. Do not include any explanations or comments, just the code.  "

    # 5. Ask the LLM to generate the code
    generated_code = llm.generate_code(
        system_prompt=system_prompt, 
        user_prompt=user_prompt,
        temperature=config["llm"].get("temperature", 0.2)
    )

    # 6. Save the output to a file so we can run it
    output_file = logs_dir / "generated_test_script.py"
    output_file.write_text(generated_code, encoding="utf-8")
    print(f"Code saved to {output_file}")

    # 7. Execute the generated code
    print("\n--- EXECUTING GENERATED CODE ---")
    result = executor.run_file(output_file)

    # 8. Print Results
    print("\n--- EXECUTION RESULTS ---")
    print(f"Success: {result.success}")
    print(f"Return Code: {result.return_code}")
    
    print("\nStandard Output:")
    print(result.stdout if result.stdout else "(None)")
    
    if result.stderr:
        print("\nStandard Error:")
        print(result.stderr)

if __name__ == "__main__":
    main()