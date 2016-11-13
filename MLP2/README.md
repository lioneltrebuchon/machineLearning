**** Useful commands ****

- Import the libraries:

module load python/2.7.6

- Submit a job on Euler:

bsub -W 08:00 -r -o OUTPUT_NAME.txt -R "rusage[mem=8000]" python SCRIPT_NAME.py
