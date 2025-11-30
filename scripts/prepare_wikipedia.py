# # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#

import os
import sys
from pathlib import Path

# Set LD_LIBRARY_PATH to include conda environment's libsndfile before importing any modules
# that depend on it (e.g., fairseq2n)
# Dynamically detect paths based on current environment
_current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

# Detect venv lib path (assumes script is run from project root or venv is at .venv)
_script_dir = Path(__file__).parent.parent
_venv_lib = _script_dir / ".venv" / "lib"
if not _venv_lib.exists():
    # Fallback: try to get from sys.executable
    _python_path = Path(sys.executable)
    if ".venv" in str(_python_path) or "venv" in str(_python_path):
        _venv_lib = _python_path.parent.parent / "lib"

# Detect conda lib path
_conda_lib = None
if "CONDA_PREFIX" in os.environ:
    _conda_lib = Path(os.environ["CONDA_PREFIX"]) / "lib"
else:
    # Try common conda locations for lcm-helper environment
    _env_name = os.environ.get("CONDA_DEFAULT_ENV", "lcm-helper")
    _username = os.environ.get("USER", os.environ.get("LOGNAME", Path.home().name))
    for _conda_base in [
        Path.home() / "miniconda3" / "envs",
        Path.home() / "miniconda" / "envs",
        Path("/u") / _username / "miniconda3" / "envs",
        Path("/u") / _username / "miniconda" / "envs",
    ]:
        _potential = _conda_base / _env_name / "lib"
        if _potential.exists() and (_potential / "libsndfile.so").exists():
            _conda_lib = _potential
            break
    # If not found with env name, try just lcm-helper
    if _conda_lib is None and _env_name != "lcm-helper":
        for _conda_base in [
            Path.home() / "miniconda3" / "envs",
            Path.home() / "miniconda" / "envs",
            Path("/u") / _username / "miniconda3" / "envs",
            Path("/u") / _username / "miniconda" / "envs",
        ]:
            _potential = _conda_base / "lcm-helper" / "lib"
            if _potential.exists() and (_potential / "libsndfile.so").exists():
                _conda_lib = _potential
                break

# Build LD_LIBRARY_PATH with only existing paths
_ld_paths = []
if _venv_lib.exists():
    _ld_paths.append(str(_venv_lib))
if _conda_lib and _conda_lib.exists():
    _ld_paths.append(str(_conda_lib))
if _current_ld_path:
    _ld_paths.append(_current_ld_path)

if _ld_paths:
    os.environ["LD_LIBRARY_PATH"] = ":".join(_ld_paths)

import asyncio
import re
from pathlib import Path

import fire
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.partitioned_data_mapper import stopes_data_mapper
from stopes.modules.preprocess.sonar_text_embedding import (
    LangColumnConfig,
    SonarTextEmbedderConfig,
)
from stopes.utils.sharding.abstract_shards import BatchFormat
from stopes.utils.sharding.parquet_shards import (
    ParquetOutputConfig,
    ParquetShardingConfig,
)

from lcm.datasets.sentence_splitter_pipeline import (
    FullPipeline,
    FullPipelineConfig,
    SentenceSplitterConfig,
)


def convert_to_natural_language(structured_text: str) -> str:
    """
    Convert structured ToM tracking output to natural language format.
    
    Args:
        structured_text: The structured output with format like:
            "Agent's belief on the object:
            - Step X: (description) {key: value; key: value; ...}"
    
    Returns:
        Natural language version of the tracking
    """
    # Extract the agent and object from the title
    lines = structured_text.strip().split('\n')
    if not lines:
        return structured_text
        
    # Parse the title line
    title_match = re.match(r"(.+)'s belief on the (.+):", lines[0])
    if not title_match:
        return structured_text
        
    agent = title_match.group(1)
    obj = title_match.group(2)
    
    natural_text = f"Let me trace {agent}'s belief about where the {obj} is located throughout the story.\n\n"
    
    for line in lines[1:]:
        line = line.strip()
        if not line or not line.startswith('- Step'):
            continue
            
        # Parse step information
        step_match = re.match(r'- Step (\d+): \(([^)]+)\) \{(.+)\}', line)
        if not step_match:
            continue
            
        step_num = step_match.group(1)
        description = step_match.group(2)
        details = step_match.group(3)
        
        # Parse the details
        detail_dict = {}
        for detail in details.split('; '):
            if ':' in detail:
                key, value = detail.split(':', 1)
                detail_dict[key.strip()] = value.strip()
        
        # Generate natural language description
        agent_loc = detail_dict.get(f'{agent} location', 'unknown')
        obj_loc = detail_dict.get(f'{obj} location', 'unknown')
        can_see = detail_dict.get(f'{agent} sees {obj}', 'False') == 'True'
        belief = detail_dict.get(f"{agent}'s belief on {obj}", 'None')
        
        if step_num == '0':
            natural_text += f"Initially, {agent} is in the {agent_loc} and the {obj} is at the {obj_loc}. "
        else:
            # Determine what actually happened this step
            agent_moved = f'{agent} location changed' in description
            obj_moved = f'{obj} location changed' in description
            
            if agent_moved and not obj_moved:
                natural_text += f"At step {step_num}, {agent} moves to the {agent_loc}. "
            elif obj_moved and not agent_moved:
                natural_text += f"At step {step_num}, the {obj} is moved to the {obj_loc}. "
            elif agent_moved and obj_moved:
                natural_text += f"At step {step_num}, {agent} moves to the {agent_loc} and the {obj} is moved to the {obj_loc}. "
            else:
                natural_text += f"At step {step_num}, {agent} and the {obj} do not move. "
        
        # Add belief information
        if belief == 'None':
            natural_text += f"{agent} cannot see the {obj} and has no belief about its location. "
        elif can_see:
            natural_text += f"{agent} can see the {obj} at the {obj_loc}, so {agent} believes it is there. "
        else:
            natural_text += f"{agent} cannot see the {obj} but still believes it is at the {belief}. "
        
        natural_text += "\n"
    
    # Add final conclusion
    final_belief_match = re.search(r'Final Answer: (.+)', structured_text)
    if final_belief_match:
        final_answer = final_belief_match.group(1)
        natural_text += f"\nTherefore, {agent} believes the {obj} is at {final_answer}."
    
    return natural_text


def prepare_longproc_split(
    split: str = "tom_tracking_0.5k",
    cache_dir: Path = Path("./preprocessed_data"),
    use_natural_language: bool = True,
    add_split_column: bool = True,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    """
    Pre-extract a single split using streaming mode and save as parquet.
    
    This avoids triggering download_and_prepare for all splits, which causes
    schema conflicts in PrincetonPLI/LongProc dataset.
    
    Args:
        split: Split name to extract (e.g., "tom_tracking_0.5k")
        cache_dir: Directory to store the parquet file
        use_natural_language: If True, convert structured output to natural language
        add_split_column: If True, add a 'split' column for train/validation partitioning
        train_ratio: Ratio of data to use for training (default: 0.9)
        seed: Random seed for split reproducibility (default: 42)
        
    Returns:
        Path to the saved parquet file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Include natural language suffix in filename to distinguish versions
    suffix = "_natural" if use_natural_language else "_structured"
    parquet_path = cache_dir / f"longproc_{split}{suffix}.parquet"
    
    if parquet_path.exists():
        print(f"✓ Using cached parquet: {parquet_path}")
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=str(parquet_path))
        if isinstance(ds, dict):
            ds = ds[list(ds.keys())[0]]
        print(f"  Cached dataset has {len(ds)} examples")
        return parquet_path
    
    print(f"Loading {split} with streaming mode...")
    print("  This avoids triggering the generation of all splits.")
    if use_natural_language:
        print("  [Natural Language Mode]: Converting structured output to natural language")
    
    from datasets import load_dataset, Dataset
    
    # Use streaming=True to avoid triggering global builder
    ds = load_dataset(
        "PrincetonPLI/LongProc",
        split=split,
        streaming=True,  
        trust_remote_code=True,
    )
    
    # Collect all data
    print("Collecting data from streaming dataset...")
    all_data = []
    converted_count = 0
    for i, item in enumerate(ds):
        # Apply natural language conversion if enabled
        if use_natural_language and 'reference_output' in item:
            original_output = item['reference_output']
            natural_output = convert_to_natural_language(original_output)
            # Only update if conversion was successful (text changed)
            if natural_output != original_output:
                item['reference_output'] = natural_output
                converted_count += 1
        
        all_data.append(item)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} samples...")
    
    # Add split column if requested
    if add_split_column:
        import random
        random.seed(seed)
        
        total_samples = len(all_data)
        train_size = int(total_samples * train_ratio)
        
        # Randomly shuffle indices
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_indices = set(indices[:train_size])
        
        for i in range(total_samples):
            if i in train_indices:
                all_data[i]['split'] = 'train'
            else:
                all_data[i]['split'] = 'validation'
        
        print(f"  Added split column: {train_size} train, {total_samples - train_size} validation")
    
    # Convert to Dataset and save as parquet
    print(f"Converting to Dataset and saving to parquet...")
    ds_final = Dataset.from_list(all_data)
    ds_final.to_parquet(parquet_path)
    
    print(f"✓ Saved {len(ds_final)} samples to {parquet_path}")
    if use_natural_language:
        print(f"  Converted {converted_count}/{len(ds_final)} reference outputs to natural language")
    return parquet_path


def run(
    output_dir: Path,
    split: str = "tom_tracking_0.5k",
    num_shards: int = 1,
    batch_size: int = 5,
    cache_dir: Path = Path("./preprocessed_data"),
    use_natural_language: bool = True,
    add_split_column: bool = True,
    train_ratio: float = 0.8,  # 统一为 0.8，与 prepare_longproc_split 保持一致
    seed: int = 42,
):
    """
    Launch a preprocessing pipeline for PrincetonPLI/LongProc tom_tracking dataset.
    This will use SAT to split text in sentences and then use SONAR to embed each sentence.
    
    This example downloads data from huggingface and outputs it to a parquet dataset.

    Args:
        output_dir: The directory where the processed data will be written. The output will be in a parquet file format.
        split: The split name to use from the dataset (e.g., "tom_tracking_0.5k", "tom_tracking_2k", "tom_tracking_8k")
        num_shards: Number of shards to split the dataset into for parallel processing
        batch_size: Batch size for processing
        cache_dir: Directory to cache the pre-extracted parquet file (default: ./preprocessed_data)
        use_natural_language: If True, convert structured reference_output to natural language format (default: True)
        add_split_column: If True, add 'split' column for train/validation partitioning (default: False)
        train_ratio: Ratio of data for training when add_split_column=True (default: 0.9)
        seed: Random seed for split reproducibility (default: 42)
    """
    # Step 1: Pre-extract the split using streaming mode and save as parquet
    # This avoids triggering download_and_prepare for all splits
    print("=" * 80)
    print("Step 1: Pre-extracting split using streaming mode")
    print("=" * 80)
    parquet_file = prepare_longproc_split(
        split=split, 
        cache_dir=cache_dir,
        use_natural_language=use_natural_language,
        add_split_column=add_split_column,
        train_ratio=train_ratio,
        seed=seed
    )
    
    # setup the sentence splitter
    # For PrincetonPLI/LongProc tom_tracking dataset, the text fields are "input_prompt" and "reference_output"
    # The dataset contains fields: instance_id, input_prompt, reference_output, item
    splitter_config = SentenceSplitterConfig(
        columns=[
            "input_prompt",      # source/prompt text
            "reference_output"   # target/expected output text
        ],  # this is the column in the input dataset where we expect to find text to split
        model_name="sat-3l",
        verbose=True,
        sentence_threshold=0.2,  # sentence splitting threshold to tune based on the data (domain, language, etc.)
        max_sentence_len=256,
    )
    # setup SONAR, we are only going to deal with english
    sonar_encoder_config = SonarTextEmbedderConfig(
        column_config=[  # we can process several columns at once which is useful for finetuning datasets
            LangColumnConfig("input_prompt_sentences", lang_value="eng_Latn"),      # source embeddings
            LangColumnConfig("reference_output_sentences", lang_value="eng_Latn"),  # target embeddings
        ],  # splitter has output new columns with "_sentences" suffix, SONAR will add "_sonar_emb" suffix
        device="cuda",  # we want to work on a GPU, if you want to try this on a cpu, change the device here
    )
    # setup the full pipeline, that will use the splitter and the sonar embeddings,
    full_config = FullPipelineConfig(
        splitter_config=splitter_config,
        sonar_encoder_config=sonar_encoder_config,
    )

    # Step 2: Use ParquetShardingConfig instead of HFInputConfig
    # This avoids the HFShard.__enter__() issue entirely
    print("\n" + "=" * 80)
    print("Step 2: Setting up input configuration using ParquetShardingConfig")
    print("=" * 80)
    print(f"Using parquet file: {parquet_file}")
    
    # Use ParquetShardingConfig instead of HFInputConfig
    # This reads from the local parquet file we just created
    # Note: ParquetShardingConfig automatically creates shards based on parquet fragments
    # Use `take` parameter to limit the number of shards if needed
    input_config = ParquetShardingConfig(
        input_file=str(parquet_file),  # ✅ Use local parquet file instead of HF dataset
        batch_format=BatchFormat.ARROW,
        batch_size=batch_size,  # adjust to your system's size
        take=num_shards if num_shards > 1 else None,  # limit shards if specified
    )
    # setup the output to write to parquet
    output_config = ParquetOutputConfig(
        output_dir,
        keep_same_partitioning=False,
        row_group_size=200,
        batch_size=200,
    )

    # requirements for our slurm jobs, if you are using a local cpu, you can ignore this
    # if you are using slurm but no gpus, remove the gpus_per_node config
    req = Requirements(
        mem_gb=128, gpus_per_node=1, cpus_per_task=10, timeout_min=3 * 24 * 60
    )
    # launching config, here we use `local` to run locally, but you can switch it to `slurm` if you have a SLURM cluster.
    launcher = Launcher(
        cache=None,
        cluster="local",
        # for SLURM you can set some parameters of the launcher here
        # cluster="slurm",
        # update_parameters={
        #    "account": "bfaq-delta-gpu",
        #    "partition": "gpuA100x4",
        #    "gres": "gpu:1",
        #    "time": "48:00:00",
        # },
    )

    # launch the shards processing
    stopes_wrapped = stopes_data_mapper(req, {"name": "prep_tom_tracking"})(FullPipeline)
    stopes_module = stopes_wrapped(input_config, output_config, full_config)

    asyncio.run(launcher.schedule(stopes_module))


if __name__ == "__main__":
    fire.Fire(run)

