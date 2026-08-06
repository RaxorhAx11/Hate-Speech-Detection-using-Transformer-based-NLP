import os
import sys
import subprocess
import time

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.utils import setup_logging

logger = setup_logging("run_pipeline")

def run_script(script_name: str):
    logger.info(f"========== RUNNING {script_name} ==========")
    script_path = os.path.join(project_root, "scripts", script_name)
    start_time = time.time()
    
    # Run the python script
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Finished {script_name} in {elapsed:.2f} seconds.")
    
    if result.returncode != 0:
        logger.error(f"Error in {script_name} (Exit code: {result.returncode})")
        logger.error(f"Stdout:\n{result.stdout}")
        logger.error(f"Stderr:\n{result.stderr}")
        raise RuntimeError(f"Pipeline step {script_name} failed.")
    else:
        # Filter and log output
        stdout_filtered = "\n".join([line for line in result.stdout.splitlines() if "it/s" not in line])
        logger.info(f"Output:\n{stdout_filtered}")

def main():
    logger.info("=========================================")
    logger.info("   STARTING END-TO-END DATASET PIPELINE   ")
    logger.info("=========================================")
    
    pipeline_steps = [
        "download_datasets.py",
        "validate_dataset.py",
        "normalize_labels.py",
        "clean_dataset.py",
        "merge_datasets.py",
        "split_dataset.py",
        "generate_reports.py"
    ]
    
    start_time = time.time()
    
    for step in pipeline_steps:
        try:
            run_script(step)
        except Exception as e:
            logger.critical(f"Pipeline failed at step {step}: {e}")
            sys.exit(1)
            
    total_time = time.time() - start_time
    logger.info("=========================================")
    logger.info(f" PIPELINE COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MIN ")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
