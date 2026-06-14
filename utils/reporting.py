from pathlib import Path

class AnalysisLogger:
    def __init__(self, output_path, filename, overwrite=True):
        """Initializes the logger, creates the directory, and sets up the file."""
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True) 
        self.output_path = self.output_path / filename
        
        # Determine mode: write (overwrite) or append
        mode = "w" if overwrite else "a"
        
        with open(self.output_path, mode) as f:
            f.write("=" * 66 + "\n")
            f.write("[*] STARTING NEW LOG PROCESS\n")
            f.write("=" * 66 + "\n\n")
            
        print(f"[*] LOG FILE INITIALIZED IN {self.output_path}")

    def section(self, title):
        """Creates a clean visual break for new analysis steps."""
        divider = "=" * 66
        formatted_title = f"\n{divider}\n{title}\n{divider}"
        
        print(formatted_title) # Print to terminal
        with open(self.output_path, "a") as f:
            f.write(formatted_title + "\n\n") # Append to file

    def log(self, message, label=None):
        """Prints a step to the terminal and appends it to the log file."""
        # Use your clean <40 spacing if a label is provided
        if label:
            output_str = f"{label:<40} {message}"
        else:
            output_str = str(message)
            
        print(output_str) # Print to terminal
        with open(self.output_path, "a") as f:
            f.write(output_str + "\n") # Append to file