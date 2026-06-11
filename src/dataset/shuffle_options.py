import random
def shuffle_options_and_update_answer_CT(sample):
    """
    随机重排选项并更新对应的答案
    """
    # 定义选项标签
    option_labels = ['A', 'B', 'C', 'D']
    

    # 提取当前的问题数据
    human_value = sample['conversations'][0]['value']
    gpt_value = sample['conversations'][1]['value']
    
    # 从human_value中提取选项内容（这里需要解析字符串）
    # 由于选项在字符串中，我们需要提取它们
    lines = human_value.split('\n')
    options_start = False
    current_options = {}
    
    for line in lines:
        line = line.strip()
        if '"A":' in line:
            current_options['A'] = line.split('"A":')[1].strip().strip('",')
        elif '"B":' in line:
            current_options['B'] = line.split('"B":')[1].strip().strip('",')
        elif '"C":' in line:
            current_options['C'] = line.split('"C":')[1].strip().strip('",')
        elif '"D":' in line:
            current_options['D'] = line.split('"D":')[1].strip().strip('",')
    
    # 从gpt_value中提取当前答案
    current_answer = None
    if '"answer": "A"' in gpt_value:
        current_answer = 'A'
    elif '"answer": "B"' in gpt_value:
        current_answer = 'B'
    elif '"answer": "C"' in gpt_value:
        current_answer = 'C'
    elif '"answer": "D"' in gpt_value:
        current_answer = 'D'
    
    if not current_answer or len(current_options) != 4:
        return  sample # 跳过无法解析的样本
    
    # 获取正确答案的内容
    correct_option_content = current_options[current_answer]
    
    # 创建选项内容列表
    option_contents = [current_options[label] for label in option_labels]
    
    # 随机打乱选项
    shuffled_contents = option_contents.copy()
    random.shuffle(shuffled_contents)
    
    # 创建新的选项映射
    new_options = {}
    for i, label in enumerate(option_labels):
        new_options[label] = shuffled_contents[i]
    
    # 找到正确答案的新位置
    new_answer = None
    for label, content in new_options.items():
        if content == correct_option_content:
            new_answer = label
            break
        
    # 重构human_value
    # 提取question和condition
    question_line = [line for line in lines if '"question":' in line]
    condition_line = [line for line in lines if '"condition":' in line]
    
    if question_line and condition_line:
        question_content = question_line[0].split('"question":')[1].strip().strip('",')
        condition_content = condition_line[0].split('"condition":')[1].strip().strip('",')
        
        # 重构human_value
    #     new_human_value = f"""<image>\n<image>\nBased on the two chest X-ray (CXR) images provided, please analyze and answer the following question, providing the reasoning for your judgment.
    # Question and options:
    # {{
    # "question": "{question_content}",
    # "condition": "{condition_content}",
    # "options": {{
    #     "A": "{new_options['A']}",
    #     "B": "{new_options['B']}",
    #     "C": "{new_options['C']}",
    #     "D": "{new_options['D']}"
    # }}
    # }}
    # Your response must strictly follow the format below: "answer": "A/B/C/D", "reason": "Brief explanation of the basis for your judgment".
    # """
        new_human_value = f"""<image>\n<image>\nBased on the two CT images provided, please analyze and answer the following question, providing the reasoning for your judgment.
    Question and options:
    {{
    "question": "{question_content}",
    "options": {{
        "A": "{new_options['A']}",
        "B": "{new_options['B']}",
        "C": "{new_options['C']}",
        "D": "{new_options['D']}"
    }}
    }}
    Your response must strictly follow the format below: "answer": "A/B/C/D", "reason": "Brief explanation of the basis for your judgment".
    """
        # 提取reason内容
        reason_content = gpt_value.split('"reason":')[1].strip().strip('".').strip('"')
        
        # 重构gpt_value
        new_gpt_value = f""""answer": "{new_answer}", "reason": "{reason_content}"."""
        
        # 更新样本
        sample['conversations'][0]['value'] = new_human_value
        sample['conversations'][1]['value'] = new_gpt_value


def shuffle_options_and_update_answer_CXR(sample):
    """
    随机重排选项并更新对应的答案
    """
    # 定义选项标签
    option_labels = ['A', 'B', 'C', 'D']
    

    # 提取当前的问题数据
    human_value = sample['conversations'][0]['value']
    gpt_value = sample['conversations'][1]['value']
    
    # 从human_value中提取选项内容（这里需要解析字符串）
    # 由于选项在字符串中，我们需要提取它们
    lines = human_value.split('\n')
    options_start = False
    current_options = {}
    
    for line in lines:
        line = line.strip()
        if '"A":' in line:
            current_options['A'] = line.split('"A":')[1].strip().strip('",')
        elif '"B":' in line:
            current_options['B'] = line.split('"B":')[1].strip().strip('",')
        elif '"C":' in line:
            current_options['C'] = line.split('"C":')[1].strip().strip('",')
        elif '"D":' in line:
            current_options['D'] = line.split('"D":')[1].strip().strip('",')
    
    # 从gpt_value中提取当前答案
    current_answer = None
    if '"answer": "A"' in gpt_value:
        current_answer = 'A'
    elif '"answer": "B"' in gpt_value:
        current_answer = 'B'
    elif '"answer": "C"' in gpt_value:
        current_answer = 'C'
    elif '"answer": "D"' in gpt_value:
        current_answer = 'D'
    
    if not current_answer or len(current_options) != 4:
        return  sample # 跳过无法解析的样本
    
    # 获取正确答案的内容
    correct_option_content = current_options[current_answer]
    
    # 创建选项内容列表
    option_contents = [current_options[label] for label in option_labels]
    
    # 随机打乱选项
    shuffled_contents = option_contents.copy()
    random.shuffle(shuffled_contents)
    
    # 创建新的选项映射
    new_options = {}
    for i, label in enumerate(option_labels):
        new_options[label] = shuffled_contents[i]
    
    # 找到正确答案的新位置
    new_answer = None
    for label, content in new_options.items():
        if content == correct_option_content:
            new_answer = label
            break
        
    # 重构human_value
    # 提取question和condition
    question_line = [line for line in lines if '"question":' in line]
    condition_line = [line for line in lines if '"condition":' in line]
    
    if question_line and condition_line:
        question_content = question_line[0].split('"question":')[1].strip().strip('",')
        condition_content = condition_line[0].split('"condition":')[1].strip().strip('",')
        
        # 重构human_value
        new_human_value = f"""<image>\n<image>\nBased on the two chest X-ray (CXR) images provided, please analyze and answer the following question, providing the reasoning for your judgment.
    Question and options:
    {{
    "question": "{question_content}",
    "options": {{
        "A": "{new_options['A']}",
        "B": "{new_options['B']}",
        "C": "{new_options['C']}",
        "D": "{new_options['D']}"
    }}
    }}
    Your response must strictly follow the format below: "answer": "A/B/C/D", "reason": "Brief explanation of the basis for your judgment".
    """
        
        # 提取reason内容
        reason_content = gpt_value.split('"reason":')[1].strip().strip('".').strip('"')
        
        # 重构gpt_value
        new_gpt_value = f""""answer": "{new_answer}", "reason": "{reason_content}"."""
        
        # 更新样本
        sample['conversations'][0]['value'] = new_human_value
        sample['conversations'][1]['value'] = new_gpt_value