# DONE: Distributed Newton Method for Federated Edge Learning (IEEE Transactions on Parallel and Distributed Systems 2022).

This repository is for the Experiment Section of the paper:
"DONE: Distributed Newton Method for Federated Edge Learning"

Link: https://arxiv.org/pdf/2012.05625.pdf, https://ieeexplore.ieee.org/abstract/document/9695269

# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc.

# Dataset: We use 2 datasets: MNIST, and Synthetic

- To generate non-idd MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_32users.py"
  - We can change the number of user and number of labels for each user using 2 variable NUM_USERS = 32 and NUM_LABELS = 3
  - 
- The datasets also are available to download at: https://drive.google.com/drive/folders/1LkBjkP0PzfRNiAY9ImN85r9vBIuW4U6-?usp=sharing

# Produce experiments and figures
- There is a main file  "plot.py" to plot all results after runing all experiment.  

## Performance comparison with different distributed algorithms (table 2 in our paper)
- For MNIST:
      <pre><code>
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm Newton --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm Newton --batch_size 0 --alpha 0.05 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DANE --batch_size 0 --eta 1 --learning_rate 0.04 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm FEDL --batch_size 0 --eta 1 --learning_rate 0.04 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm GD --batch_size 0 --learning_rate 0.2 --num_global_iters 100 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm GT --batch_size 0 --alpha 0.03 --L 1 --num_global_iters 100 --local_epochs 40 --numedges 32 
    </code></pre>
- For FEMNIST:
      <pre><code>
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm Newton --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm Newton --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm DANE --batch_size 0 --eta 1 --learning_rate 0.02 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm FEDL --batch_size 0 --eta 1 --learning_rate 0.02 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm GD --batch_size 0 --learning_rate 0.02 --num_global_iters 100 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm GT --batch_size 0 --alpha 0.01 --L 1 --num_global_iters 100 --local_epochs 40 --numedges 32
    </code></pre>

- For Human Activities:
      <pre><code>
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm Newton --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm Newton --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm DANE --batch_size 0 --eta 1 --learning_rate 0.05 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm FEDL --batch_size 0 --eta 1 --learning_rate 0.05 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm GD --batch_size 0 --learning_rate 0.1 --num_global_iters 100 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm GT --batch_size 0 --alpha 0.01 --L 1 --num_global_iters 100 --local_epochs 40 --numedges 30
    </code></pre>
此项目转载自https://github.com/CharlieDinh/DONE
