import re
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel
from pathlib import Path
import fire
from loguru import logger
from tqdm import tqdm
import srsly

import torch
from lsdist.utils import get_env, get_script_id, get_ver
from lsdist.utils.io import use_file
from lsdist.utils.numerical import init_random

DEV = get_env() == "development"
SCRIPT_ID = get_script_id(__file__) + get_ver()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHUFFLE_SEED = 20231005


def main(
    scalar_name: str = "max_bin100",
    split_name: str = "bench",
    oai_fn: str = f"saved/ssch/export_openai_format/my_{{scalar_name}}_{{split_name}}.json",
    ssch_json: str = f"saved/{SCRIPT_ID}/my_{{scalar_name}}_{{split_name}}.json",
    batch_size: int = 8,
    ft_names: tuple[str] = ("lora",),
    num_values: int = 1,
):
    init_random()
    logger.info(f"{SCRIPT_ID=}, {locals()=}")

    # Parameter
    oai_fn: Path = use_file(
        oai_fn.format(scalar_name=scalar_name, split_name=split_name), new=False
    )
    ssch_json: Path = use_file(
        ssch_json.format(scalar_name=scalar_name, split_name=split_name), new=True
    )

    # https://llama.meta.com/docs/how-to-guides/fine-tuning/

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    new_model = f"saved/ssch/llama2_lora_sft/{scalar_name}"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    if "lora" in ft_names:
        model = PeftModel.from_pretrained(base_model, new_model)
    else:
        model = base_model

    # Note that the passed model may be modified inplace.
    # model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        do_sample=False,
        top_p=None,
        temperature=None,
        batch_size=batch_size,
    )

    oai_lst = srsly.read_json(oai_fn)
    inputs_lst = []
    for oai_item in oai_lst:
        prompt = oai_item["messages"][0]["content"]
        inputs_lst.append(f"<s>[INST] {prompt} [/INST] ")
        # Not sure space is needed

    def find_last_number(text):
        all_numbers = re.findall(r"\d+", text)
        if all_numbers:
            if num_values == 1:
                last_number = all_numbers[-1]
                if len(last_number) <= 6:
                    return int(last_number)
            else:
                last_number_lst = []
                for n in all_numbers[-num_values:]:
                    if len(n) <= 6:
                        last_number_lst.append(int(n))
                if len(last_number_lst) == num_values:
                    return last_number_lst
        return None

    ssch_lst = [oai_item.copy() for oai_item in oai_lst]
    for name in ft_names:
        bad_predict_lst = []
        bad_retry_lst = []
        for i in tqdm(range(0, len(inputs_lst), batch_size), desc=f"{name}"):
            time_start = time.time()
            chunk = inputs_lst[i : i + batch_size]
            if name == "base" and "lora" in ft_names:
                with model.disable_adapter():
                    result = pipe(chunk)
            else:
                result = pipe(chunk)
            time_cost = time.time() - time_start
            for j in range(len(chunk)):
                ssch_item = ssch_lst[i + j]
                ssch_item[f"{name}_response_time"] = time_cost

                generated_text: str = result[j][0]["generated_text"]
                re_result = find_last_number(generated_text)
                retry_count = 0
                for _ in range(6):
                    if re_result is not None:
                        break
                    retry_count += 1
                    _temperature = retry_count * 0.1
                    logger.warning(
                        f"Retry {name=}, {ssch_item['dataset_idx']=}, {retry_count=}, {generated_text=}, {_temperature=}"
                    )
                    generated_text: str = pipe(
                        chunk[j], do_sample=True, top_p=0.9, temperature=_temperature
                    )[0]["generated_text"]
                    re_result = find_last_number(generated_text)

                ssch_item[f"{name}_response_predict"] = generated_text
                ssch_item[f"{name}_num_response_predict"] = re_result
                ssch_item[f"{name}_retry_count"] = retry_count
                if re_result is None:
                    bad_retry_lst.append(ssch_item["dataset_idx"])
                    logger.warning(f"Bad {name=}, {ssch_item['dataset_idx']=}, {generated_text=}")
                # print(srsly.json_dumps(ssch_lst[1], indent=4))

            if i % 500 == 0:
                srsly.write_json(ssch_json, ssch_lst)
        srsly.write_json(ssch_json, ssch_lst)

        logger.info(f"{name=} done!")
        if bad_predict_lst:
            logger.warning(f"{name=}, {bad_predict_lst=}")
        if bad_retry_lst:
            logger.error(f"{name=}, {bad_retry_lst=}")


if __name__ == "__main__":
    fire.Fire(main)
