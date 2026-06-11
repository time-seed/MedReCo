import torch
import deepspeed
import json
from src.model.modeling_qwen2_5_vl_ctvit import Qwen2_5_VLForConditionalGeneration
from src.model.processing_qwen2_5_vl_ctvit import Qwen2_5_VLProcessor
from src.dataset.vision_process_vit import process_vision_info

modality = 'CTRATE'
class_ = 'Abnormality'
type_ = 'free'


def main():
    # 设置单 GPU
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')

    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        '/mnt/petrelfs/zhangtengfei/public_dataset/Diff_dataset/pubilc_diff_dataset/our_527/qwen_result',
        device_map={"": 0},
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(
        "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_v2/new_temp_config"
    )

    # 初始化 DeepSpeed 推理引擎
    model = deepspeed.init_inference(
        model,
        mp_size=1,
        dtype=torch.bfloat16,
        checkpoint=None,
        replace_with_kernel_inject=True,
        replace_method="auto",
        max_tokens=1500
    )
    model.eval()

    # 加载数据，只取第一条
    with open(f'/mnt/petrelfs/zhangtengfei/public_dataset/Diff_dataset/Diff_test_dataset/Diff_Data/{modality}/{class_}/{modality.lower()}_{class_.lower()}_test_{type_}.jsonl') as infile:
        data = json.load(infile)

    entry = data[0]                 # 只跑这一条；想换别的就改索引
    sample_id = entry['id']
    messages = [entry]
    print(entry)
    # 构造输入
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

    # 生成输出
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False
    # )[0]
    output_text = processor.tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"ID: {sample_id}")
    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()