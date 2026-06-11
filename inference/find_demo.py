"""
select_case.py
----------------
从测试集中挑选 1 个 case，做三件事：
  1. 把这个 case 用到的图片复制到 inference/examples/（命名为 case_A / case_B ...）
  2. 把图片路径改写成本地相对路径，存成 inference/examples/example.json（供 inference.py 使用）
  3. 打印出这个 case 的问题（question）

这个脚本只是你本地准备 demo 用的，最终不需要开源。
最终开源的只有：inference.py + inference/examples/ 里的两张图片(+example.json)。

用法（从仓库根目录运行）：
    python inference/select_case.py                 # 默认取第 0 个样本
    python inference/select_case.py --index 5       # 取第 5 个样本
    python inference/select_case.py --id "xxxxxx"   # 按 id 取样本
"""

import os
import json
import shutil
import argparse

# ====== 和原推理脚本保持一致的配置 ======
modality = 'MIMIC'
class_ = 'Anatomy'      # 可选: 'Abnormality' / 'Anatomy' / 'Disease'
type_ = 'free'          # 可选: 'free' / '1113'

# 原始测试集路径（按需修改）
DATA_PATH = (
    f'/mnt/petrelfs/zhangtengfei/public_dataset/Diff_dataset/Diff_test_dataset/'
    f'Diff_Data/{modality}/{class_}/'
    f'{modality.lower()}_{class_.lower()}_test_{type_}.jsonl'
)

# 输出目录（demo 用）
EXAMPLES_DIR = 'inference/examples'


def extract_images(content):
    """从一条 message 的 content 中找出所有图片项 (item, 原始路径)"""
    images = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'image' and item.get('image'):
                images.append((item, item['image']))
    return images


def extract_question(content):
    """取出文本问题"""
    if isinstance(content, list):
        texts = [it.get('text', '') for it in content
                 if isinstance(it, dict) and it.get('type') == 'text']
        return '\n'.join(t for t in texts if t)
    if isinstance(content, str):
        return content
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='选第几个样本')
    parser.add_argument('--id', type=str, default=None,
                        help='按 id 选样本（优先级高于 --index）')
    args = parser.parse_args()

    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)

    # 选 case
    if args.id is not None:
        entry = next(e for e in data if str(e.get('id')) == args.id)
    else:
        entry = data[args.index]

    # 深拷贝，避免改到原始数据对象
    entry = json.loads(json.dumps(entry))

    content = entry['content']
    images = extract_images(content)

    if not images:
        print('警告：没有在该样本里找到图片项，请检查数据格式。')

    # 复制图片，命名为 case_A / case_B ...，并改写 message 里的路径为相对路径
    # （id 里带斜杠和括号，不能直接做文件名，所以这里用字母标号）
    labels = [chr(ord('A') + i) for i in range(len(images))]
    for i, (item, src) in enumerate(images):
        ext = os.path.splitext(src)[1] or '.jpg'
        dst_name = f'case_{labels[i]}{ext}'              # case_A.jpg, case_B.jpg
        dst_path = os.path.join(EXAMPLES_DIR, dst_name)
        shutil.copy(src, dst_path)

        # 改写为相对 inference/ 的路径，inference.py 里会再解析成绝对路径
        item['image'] = os.path.join('examples', dst_name)
        print(f'已复制图片 (Case {labels[i]}): {src}  ->  {dst_path}')

    # 保存这条样本，供 inference.py 加载
    out_json = os.path.join(EXAMPLES_DIR, 'example.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    print('\n========== Case ID ==========')
    print(entry.get('id'))
    print('========== 问题 (Question) ==========')
    print(extract_question(content))
    print('=====================================')
    print(f'\n已保存样本到: {out_json}')
    print(f'共复制图片 {len(images)} 张到 {EXAMPLES_DIR}/')


if __name__ == '__main__':
    main()