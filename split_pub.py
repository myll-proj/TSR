import jsonlines
import random  # 确保导入 random 模块
import json
import os
import cv2

# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

# 打印JSONL文件内容
def print_json(data):
    for json_obj in data:
        json_formatted_str = json.dumps(json_obj, indent=4, ensure_ascii=False)
        print(json_formatted_str)
        
# 打乱数据集并随机提取
def random_extract(train_data, num_samples):
    # 打乱数据
    random.shuffle(train_data)
    # 提取前num_samples个数据
    return train_data[:num_samples]

  
train_data = []
val_data_test = []
with jsonlines.open("/your_path/pubtabnet/PubTabNet_2.0.0.jsonl", "r") as f:    
    for item in f: # 正常使用进行
        if item['split'] == 'train':
            train_data.append(item)
        else:
            val_data_test.append(item)
    
    # 先整20个数据先，测试划分是不是对的，自己整理一下是不是和原版的内容对上的；
    random_train_data_test = random_extract(train_data, 20)

    
print('train length:',len(train_data))
print('val length:',len(val_data_test)) # 9115

# 训练集的提取
with jsonlines.open("/your_path/pubtabnet_part/PubTabNet_2.0.0_part.jsonl", "w") as train_f:
    for data in random_train_data_test:
        if data['split'] == 'train':
            
            filename = data["filename"]
            print(f"训练集*****train-data['imgid']: {data['imgid']}*****")
            train_f.write(data) # 将训练数据写入文件
            
            path = "/your_path/pubtabnet/train"
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"无法读取图片: {img_path}")
            else:
                cv2.imwrite(os.path.join("/your_path/pubtabnet_part/train", filename), img)  
            
# 验证集的提取
with jsonlines.open("/your_path/pubtabnet_part/PubTabNet_2.0.0_part.jsonl", "a") as val_f:          
    for data in val_data_test:
        if data['split'] == 'val':
            
            print(f"=====val-data['imgid']: {data['imgid']}=====")
            val_f.write(data) # 将验证数据写入文件
            
            path = "/your_path/pubtabnet/val"
            img_path = os.path.join(path, data["filename"])
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"无法读取图片: {img_path}")
            else:
                cv2.imwrite(os.path.join("/your_path/pubtabnet_part/val", data["filename"]), img) 

# 再次验证划分情况
idx_train = 0
idx_val = 0
idx_other = 0
with jsonlines.open("/your_path/pubtabnet_part/PubTabNet_2.0.0_part.jsonl", "r") as f:
    for data in f:
        if data['split'] == 'train':
            idx_train += 1
        elif data['split'] == 'val':
            idx_val += 1
        else:
            idx_other += 1

print(f"划分的训练集有：{idx_train}条")
print(f"划分的验证集有：{idx_val}条")
print(f"其他情况有：{idx_other}条") # 0条
    