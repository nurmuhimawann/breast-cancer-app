import subprocess
from pathlib import Path

# Get the current directory
current_directory = Path(__file__).resolve().parent

# Path to main.py
main_script_path = current_directory / "app" / "main.py"

# Path to activate.bat (virtual environment activation script)
activate_cmd = current_directory / "venv" / "Scripts" / "activate.bat"

# Run the command
subprocess.run(f"call {str(activate_cmd)} && streamlit run {str(main_script_path)}", shell=True)
