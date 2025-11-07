import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch.backends.cudnn as cudnn

from datasets import load_dataset

from llama_generator import LlamaGenerator
from general_prm import GeneralPRM
from mcts_openstrawberry import MCTS

import yaml
import json
from pathlib import Path
from datetime import datetime

import logging

# Create a logger object
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Decoding using Hugging Face Transformers")

# Setup
parser.add_argument('--seed', type=int, default=123)
parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for authentication.")
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--version", type=str, default=None)

parser.add_argument('--dataset', type=str)
parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum number of new tokens to generate.")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Name of the model to use.")
parser.add_argument("--use_past_key_values", type=bool, default=False, help="Whether to use past key values for faster inference.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model (e.g., cuda, cpu).")
parser.add_argument("--secondary_device", type=str, default="cpu", help="Secondary device to offload computation (e.g., cpu).")
parser.add_argument("--branch_factor", type=int, default=None, help="Branch factor for any applicable algorithm.")

# Search config
parser.add_argument("--branch_factor_tree", type=int, default=None)
parser.add_argument("--n_iters", type=int, default=None)
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--score_aggregation", type=str, default=None)

# PRM config
parser.add_argument("--prm_model_name", type=str, default="UW-Madison-Lee-Lab/VersaPRM", help="Name of the model to use.")
parser.add_argument("--positive_tag", type=str, default="+", help="Positive tag used in the model.")
parser.add_argument("--negative_tag", type=str, default="-", help="Negative tag used in the model.")
parser.add_argument("--score_token", type=str, default=" \n\n\n\n", help="Token used to calculate or indicate scores.")

parser.add_argument('--test_sample_num', type=int, default=-1)
parser.add_argument(
    '--test_sample_idx',
    type=int,
    nargs='+',  # 여러 값 허용
    default=[],
    help="실행할 샘플 인덱스 리스트 (예: --test_sample_idx 1 5 7)"
)

logging.basicConfig(level=logging.INFO)

SUPPORTING_API = ['gpt-4o-mini']

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)



dataset_query_key = {
    'openai/gsm8k' : 'question',
    'Idavidrein/gpqa' : 'Question',
    'Maxwell-Jia/AIME_2024' : 'Problem',
    'HuggingFaceH4/MATH-500': 'problem',
    'deepmind/aqua_rat' : 'question',
    'ChilleD/SVAMP' : 'question_concat',
}   


def save_config_and_prepare_dir(args):
    # 1) collect env vars
    env_vars = {
        key: os.environ.get(key)
        for key in ("CUDA_VISIBLE_DEVICES", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        if key in os.environ
    }

    # 2) build full config dict
    cfg = vars(args).copy()
    cfg.update(os.environ)

    if args.dataset == 'HuggingFaceH4/MATH-500':
        pretty_dataset = 'math500'
    else:
        raise KeyError()
    # 3) make a timestamped folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ('gpt' in args.model_name) or (args.model_name in SUPPORTING_API):
        base = Path(f"experimentsMTCS-{pretty_dataset}-gpt")
    else:
        base = Path(f"experimentsMTCS-{pretty_dataset}")
    run_dir = base / f'{args.model_name.replace("/", "_")}-{args.prm_model_name.replace("/", "_")}-{args.method.replace("/", "_")}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # 4) dump YAML
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return run_dir

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ['ALGORITHM_VERSION'] = str(args.version)
    set_seed(args.seed)

    bnb_config = {
            'load_in_4bit': True,
            'load_in_8bit': False,
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_compute_dtype': 'float16',
            'bnb_4bit_quant_type': 'nf4',
        }

    if args.dataset == 'openai/gsm8k':
        dataset = load_dataset(args.dataset, 'main', cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test'] 
    elif args.dataset == 'Idavidrein/gpqa':
        dataset = load_dataset(args.dataset, 'gpqa_diamond', cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['train'] 
    elif args.dataset == 'Maxwell-Jia/AIME_2024':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['train'] 
    elif args.dataset == 'HuggingFaceH4/MATH-500':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']     
    elif args.dataset == 'deepmind/aqua_rat':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']     
    elif args.dataset == 'ChilleD/SVAMP':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']     
    elif args.dataset == 'TIGER-Lab/MMLU-Pro':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']     
    elif 'cais/mmlu' in args.dataset:
        assert '-' in args.dataset
        dataset = load_dataset('cais/mmlu', args.dataset.split('-')[-1], cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']
    else:
        raise KeyError()

    generator = LlamaGenerator(
        max_new_tokens=args.max_new_tokens,
        model_name=args.model_name,
        quantization_config=bnb_config,
        hf_token=args.hf_token,
        use_past_key_values=args.use_past_key_values,
        batch_size=args.batch_size,
        device=args.device,
        secondary_device=args.secondary_device,
    )

    prm = GeneralPRM(
        quantization_config=bnb_config,
        model_name=args.prm_model_name,
        positive_tag=args.positive_tag,
        negative_tag=args.negative_tag,
        score_token=args.score_token,
        hf_token=args.hf_token,
        use_past_key_values=args.use_past_key_values,
        batch_size=args.batch_size,
        device=args.device,
        secondary_device=args.secondary_device,
        dtype=torch.float32,
    )
    search = MCTS(
        method=args.method,
        generator = generator, 
        prm = prm,
        branching_factor=args.branch_factor_tree,
        n_iters=args.n_iters,
        max_depth=args.max_depth,
        score_aggregation=args.score_aggregation,
    )

    results = defaultdict(dict)
    indices_to_run = range(len(test_dataset))
    if args.test_sample_idx:  # 하나라도 있으면
        indices_to_run = args.test_sample_idx

    for test_idx in tqdm(indices_to_run):
        test_sample = test_dataset[test_idx]
        if args.dataset == 'HuggingFaceH4/MATH-500':
            qid = test_idx

            # query 만들기question
            test_query = test_sample["problem"]
            formatted_query = f"{test_query}"
        elif args.dataset == 'TIGER-Lab/MMLU-Pro':
            qid = test_sample["question_id"]  # MMLU-Pro 전용

            # query 만들기
            test_query = test_sample["question"]
            options = test_sample["options"]
            formatted_query = f"{test_query}\n"
            for i, choice in enumerate(options):
                formatted_query += f"{i}. {choice}\n"
        else:
            raise KeyError()

        try:
            outputs = search(formatted_query)
            result = {qid: outputs}

            # 개별 저장
            run_dir = save_config_and_prepare_dir(args)
            fname = f"qid{qid}.json"
            with open(run_dir /fname, "w") as f:
                json.dump(result, f, indent=4)

        except Exception as err:
            logger.error(f"Error on qid {qid}: {err}")
            continue