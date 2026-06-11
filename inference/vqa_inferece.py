"""
inference.py
----------------
Single-case inference demo.
Paste your own case (images + question) into `messages` below, then run:

    python inference/inference.py
"""

import torch

from src.model.modeling_qwen2_5_vl_ctvit import Qwen2_5_VLForConditionalGeneration
from src.model.processing_qwen2_5_vl_ctvit import Qwen2_5_VLProcessor
from src.dataset.vision_process_vit import process_vision_info


MODEL_PATH = '/mnt/petrelfs/zhangtengfei/public_dataset/Diff_dataset/pubilc_diff_dataset/our_527/qwen_result',   # TODO: model weights path
PROCESSOR_PATH = "configs"


# ====== Put your case here ======
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "inference/examples/vqa_image1.jpg"},
            {"type": "image", "image": "inference/examples/vqa_iamge2.jpg"},
            {"type": "text", "text": "Comparing the two cases, how does the endotracheal tube tip's position relative to the carina differ between Case A and Case B?"},   # TODO: paste the question here
        ],
    }
]
# ================================


def main():
    device = torch.device("cuda:0")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    processor = Qwen2_5_VLProcessor.from_pretrained(PROCESSOR_PATH)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(output_text)


if __name__ == "__main__":
    main()