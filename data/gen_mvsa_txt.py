import json
import os
from tqdm import tqdm
if __name__ == "__main__":

    mvsa_dir = "/data1/zhangxiaohui/MVSA_Single"

    all_jsonls = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    result = []
    for jsonl in all_jsonls:
        
        json_path = os.path.join(mvsa_dir, jsonl)

        img_list = [json.loads(line)["img"].split("/")[-1] for line in open(json_path)]
        label_list = [json.loads(line)["label"] for line in open(json_path)]

        for i, img in tqdm(enumerate(img_list)):
            result.append("{} {}\n".format(img, label_list[i]))
        
        with open("my_{}_mvsa.txt".format(jsonl.split(".jsonl")[0]), "w") as mf:
            mf.writelines(result)
        result = []




