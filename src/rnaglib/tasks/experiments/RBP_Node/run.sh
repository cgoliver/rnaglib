#!/bin/bash -l

# Function to submit a job
submit_job() {
    local job_name=$1
    shift  # Remove the first argument (job_name) from the list of arguments
    local python_args=$@  # All remaining arguments will be passed to the Python script
    
    sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=${job_name}
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:h100_pcie_2g.20gb:1
#SBATCH --time=20-00:00:00
#SBATCH --partition=p.hpcl91
#SBATCH --output=/fs/pool/pool-wyss/RNA/rnaglib/src/rnaglib/tasks/experiments/slurm-%j.out

source /fs/home/wyss/.bashrc 
mamba activate RNA
export CUDA_VISIBLE_DEVICES=0   

python run.py ${python_args}

EOT

    echo "Submitted job: ${job_name}"
    echo "Python arguments: ${python_args}"
}

# Experiment 1: 0 layers
submit_job "RGCN_Site_0layers" --run_name "RGCN_0layers" --layers 0 --root "RBP-Node0" --experiment_name "RBP-Node" --task "BenchmarkProteinBindingSiteDetection"

# Experiment 2: 1 layer
submit_job "RGCN_1layer" --run_name "RGCN_1layer" --layers 1 --root "RBP-Node1" --experiment_name "RBP-Node" --task "BenchmarkProteinBindingSiteDetection"

# Experiment 3: 2 layers
submit_job "RGCN_2layers" --run_name "RGCN_2layers" --layers 2 --root "RBP-Node2" --experiment_name "RBP-Node" --task "BenchmarkProteinBindingSiteDetection"

# Experiment 4: 0 layers embeddings
submit_job "RGCN_0layers_emb" --run_name "RGCN_0layers_emb" --layers 0 --root "RRBP-Node0emb" --experiment_name "RBP-Node" --task "BenchmarkProteinBindingSiteDetectionEmbeddings"

# Experiment 5: 1 layer embeddings
submit_job "RGCN_1layer_emb" --run_name "RGCN_1layer_emb" --layers 1 --root "RBP-Node1emb" --experiment_name "RBP-Node" --task "BenchmarkProteinBindingSiteDetectionEmbeddings"

# Experiment 6: 2 layers embeddings
submit_job "RGCN_2layers_emb" --run_name "RGCN_2layers_emb" --layers 2 --root "RBP-Node2emb" --experiment_name "RBP-Node" --task "BenchmarkProteinBindingSiteDetectionEmbeddings"

echo "All jobs submitted"