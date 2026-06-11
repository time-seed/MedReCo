# 检查要使用到的所有条件，是否都在index json文件中出现过了

import json

# 你提供的列表（作为 JSON 格式的字符串）
list_str = '["Cardiac chamber enlargement (atrial or ventricular enlargement) in Cardiac Chambers", "Pleural effusion (including localized effusion) in Pleura", "Atelectasis in Pulmonary Parenchyma", "Lung consolidation (e.g., high-density shadow due to inflammation) in Pulmonary Parenchyma", "Interstitial fibrosis and thickening (e.g., reticular changes) in Pulmonary Interstitium", "Venous dilatation in Pulmonary Vasculature", "Atherosclerotic plaque of the arterial wall in Aorta", "Bone fracture in Ribs", "Abnormal arterial course in Aorta", "Pleural thickening in Pleura", "Ground-glass opacity (e.g., slightly high-density shadow due to inflammation) in Pulmonary Parenchyma", "Eventration of the diaphragm in Diaphragm", "Arterial dilatation in Pulmonary Vasculature", "Bone deformity in Thoracic spine", "Lymph node enlargement in the mediastinum in Mediastinum", "Pulmonary fibrosis in Pulmonary Parenchyma", "Venous dilatation in Thoracic veins", "Diffusely distributed, multiple pulmonary nodules and masses (sarcoidosis, metastasis, pneumoconiosis) in Pulmonary Parenchyma", "Mediastinal shift in Mediastinum", "Callus and post-fracture healing changes in Ribs", "Solitary pulmonary nodule and mass (peripheral lung cancer, granuloma) in Pulmonary Parenchyma", "Bone fracture in Thoracic spine", "Bone fracture in Sternum", "Soft tissue density lesions in the mediastinum (including mediastinal tumors) in Mediastinum", "Arterial dilatation in Aorta", "Bone deformity in Ribs", "Bronchiectasis in Bronchi (Lower Respiratory Tract)", "Pericardial effusion in Pericardium", "Cardiac wall hypertrophy (e.g., myocardial hypertrophy) in Cardiac Walls", "Hernia or abnormal course of the gastrointestinal tract in Stomach", "Pneumothorax in Pleura", "Pneumomediastinum in Mediastinum", "Valvular calcification in Cardiac Valves", "Incomplete lung expansion (e.g., due to pneumothorax) in Pulmonary Parenchyma"]'
# 你提供的字典
my_dict = {
        "Bone deformity in Ribs": 0,
        "Incomplete lung expansion (e.g., due to pneumothorax) in Pulmonary Parenchyma": 1,
        "Callus and post-fracture healing changes in Ribs": 2,
        "Bronchiectasis in Bronchi (Lower Respiratory Tract)": 3,
        "Mediastinal shift in Mediastinum": 4,
        "Pleural effusion (including localized effusion) in Pleura": 5,
        "Diffusely distributed, multiple pulmonary nodules and masses (sarcoidosis, metastasis, pneumoconiosis) in Pulmonary Parenchyma": 6,
        "Bronchial wall thickening in Bronchi (Lower Respiratory Tract)": 7,
        "Abnormal arterial course in Aorta": 8,
        "Emphysema in Pulmonary Parenchyma": 9,
        "Bone fracture in Ribs": 10,
        "Dilatation of the gastrointestinal lumen (e.g., increased content in proximal lumen due to obstruction of esophagus, stomach, or intestine) in Stomach": 11,
        "Atelectasis in Pulmonary Parenchyma": 12,
        "Arterial dilatation in Pulmonary Vasculature": 13,
        "Pneumomediastinum in Mediastinum": 14,
        "Interstitial fibrosis and thickening (e.g., reticular changes) in Pulmonary Interstitium": 15,
        "Bone fracture in Sternum": 16,
        "Lymph node enlargement in the mediastinum in Mediastinum": 17,
        "Pleural calcification in Pleura": 18,
        "Bone asymmetry in Thoracic spine": 19,
        "Cardiac chamber enlargement (atrial or ventricular enlargement) in Cardiac Chambers": 20,
        "Osteoporosis in Thoracic spine": 21,
        "Soft tissue density lesions of the pleura in Pleura": 22,
        "Eventration of the diaphragm in Diaphragm": 23,
        "Bone deformity in Thoracic spine": 24,
        "Hernia or abnormal course of the gastrointestinal tract in Stomach": 25,
        "Venous dilatation in Thoracic veins": 26,
        "Diaphragmatic hernia in Diaphragm": 27,
        "Pleural thickening in Pleura": 28,
        "Pulmonary fibrosis in Pulmonary Parenchyma": 29,
        "Valvular calcification in Cardiac Valves": 30,
        "Atherosclerotic plaque of the arterial wall in Aorta": 31,
        "Venous dilatation in Pulmonary Vasculature": 32,
        "Hyperostosis or sclerosis of bone in Thoracic spine": 33,
        "Arterial dilatation in Aorta": 34,
        "Bone fracture in Thoracic spine": 35,
        "Lung consolidation (e.g., high-density shadow due to inflammation) in Pulmonary Parenchyma": 36,
        "Pneumothorax in Pleura": 37,
        "Solitary pulmonary nodule and mass (peripheral lung cancer, granuloma) in Pulmonary Parenchyma": 38,
        "Soft tissue density lesions in the mediastinum (including mediastinal tumors) in Mediastinum": 39,
        "Pericardial effusion in Pericardium": 40,
        "Cardiac wall hypertrophy (e.g., myocardial hypertrophy) in Cardiac Walls": 41,
        "Abnormal venous course in Thoracic veins": 42,
        "Ground-glass opacity (e.g., slightly high-density shadow due to inflammation) in Pulmonary Parenchyma": 43,
        "Pulmonary Parenchyma": 44,
        "Cardiac Valves": 45,
        "Diaphragm": 46,
        "Cardiac Chambers": 47,
        "Bronchi (Lower Respiratory Tract)": 48,
        "Sternum": 49,
        "Ribs": 50,
        "Pulmonary Interstitium": 51,
        "Stomach": 52,
        "Aorta": 53,
        "Pleura": 54,
        "Pulmonary Vasculature": 55,
        "Trachea_Tracheal Lumen": 56,
        "Mediastinum": 57,
        "Pneumonia": 64,
        "Aortic aneurysm": 59,
        "Aortosclerosis": 60,
        "Aortic valve disease": 61,
        "Cardiomyopathy": 62,
        "Pulmonary hypertension": 63,
        "Pulmonary tuberculosis": 65,
        "Pneumomediastinum": 66,
        "Skeletal deformities": 67
    }

# 1. 将字符串解析为 Python 的 list 对象
my_list = json.loads(list_str)

# 2. 使用列表推导式找出不在字典中的元素
# item not in my_dict 默认检查的是字典的键 (keys)
missing_elements = [item for item in my_list if item not in my_dict]

# 3. 打印结果
if len(missing_elements) == 0:
    print("检查完毕：列表中的所有元素都在字典中出现了！")
else:
    print(f"检查完毕：发现有 {len(missing_elements)} 个元素没有在字典中出现。它们是：")
    for elem in missing_elements:
        print(f" - {elem}")