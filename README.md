# Code for "Optimizing LLM Throughput with Probabilistic Length Modeling and Dynamic Scheduling"

**Note:** The length samples will be provided soon for a quick start. Some organizational work is still ongoing.

## Installation

```bash
git clone git@github.com:PillarsZhang/probabilistic-optimizing-llm-throughput.git
cd probabilistic-optimizing-llm-throughput
conda env create -f environment.yml
conda activate lsdist_env
pip install -e .
```

## Dataset Preparation and Length Distribution Collection

We use the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset, which contains 52,000 instruction-following examples. The following sample code automatically downloads the dataset when run for the first time.

1. **Collect Hidden States and Length Distribution from LLM**
   Randomly select 10,000 samples from Alpaca, input prompts into the LLM, and generate responses while collecting the hidden states of the prompts:

   ```bash
   torchrun \
       --nnodes=1 \
       --nproc_per_node=1 \
       --rdzv-backend=c10d \
       --rdzv-endpoint=localhost:0 \
   scripts/sample/llama_chat.py
   ```

2. **Clean and Split Dataset**
   Process the collected data and split it into training, validation, and test sets:

   ```bash
   python scripts/sample/dist_to_trainable.py
   ```

> In future work, we will store the hidden state and length samples in safetensor and msgpack respectively, making the code more lightweight and easier to migrate.

## Building the Probabilistic Length Predictor

1. **Training the Predictor**
   Train the probabilistic length predictor:

   ```bash
   python scripts/train/pt_train.py --comment=prob
   ```

2. **Testing and Evaluation**
   The script automatically selects the checkpoint with the lowest validation loss and predicts length distributions on the test set for evaluation and visualization:

   ```bash
   python scripts/train/pt_test.py --comment=prob
   ```

> By default, the script saves checkpoints to the `saved/` directory. The best-performing checkpoint has been uploaded to the GitHub releases. Please extract it and place the `public/` folder in the root directory of the repository.

## Building Other Scalar-Based Predictors (Baselines)

1. **MLP-Based Length Predictor**
   Train and test an MLP-based model to predict the maximum response length:

   ```bash
   python scripts/train/pt_train_scalar.py \
       --quantile=max \
       --comment=max

   python scripts/train/pt_test_scalar.py \
       --quantile=max \
       --pick-comment=max
   ```

2. **Instruction Tuning for Length Prediction**

   Convert the dataset into a format suitable for instruction tuning:

   ```bash
   python scripts/ssch/export_openai_format.py --scalar-rule="('max',)"
   ```

   Use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune the LLM with LoRA:

   ```bash
   llamafactory-cli train scripts/ssch/llama2_lora_sft_max_bin100.yaml
   ```

   After training, generate predictions for the maximum response length using the tuned model:

   ```bash
   python scripts/ssch/get_ssch_predict.py \
       --scalar-name="max_bin100" \
       --split-name="bench" \
       --num-values=1 \
       --ft-names="('lora',)"
   ```


## LLM Inference Throughput Benchmark

1. **Set Common Inference Parameters**
   Define the dataset slice and the torchrun command template:

   ```bash
   dataset_selected_slices="12048:20240"
   torchrun_cmd="torchrun \
       --nnodes=1 \
       --nproc_per_node=1 \
       --rdzv-backend=c10d \
       --rdzv-endpoint=localhost:0 \
   scripts/dispatch/throughput_benchmark.py \
       --chunk-size=256 \
       --dist-batch-size=16 \
       --gen-batch-size=16 \
       --batched-rule=\"('ssch',0.6,100,vbs,fcr,)\" \
       --dataset-selected-slices=${dataset_selected_slices}"
   ```

2. **Throughput Benchmarking**

   - **Ours (VBS+FCR):**

     ```bash
     $torchrun_cmd \
         --pt-ckpt=public/train/pt_train/pt_ckpt \
         --tag='prob'
     ```

   - **MLP (VBS+FCR):**

     ```bash
     $torchrun_cmd \
         --model-scalar \
         --model-scalar-tag=max \
         --pt-ckpt=public/train/pt_train_scalar/pt_ckpt \
         --tag='max'
     ```

   - **Instruction Tuning (VBS+FCR):**

     ```bash
     $torchrun_cmd \
         --ssch-lengths-name="max_bin100" \
         --ssch-lengths-key="lora"
     ```
