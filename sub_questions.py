import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from ast import literal_eval
from tqdm import tqdm
import pandas as pd
import numpy as np
import random,pickle

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(888)

model_id = "Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)



val_annotations_path="/home/lh/llama2-okvqa/annotations/ok_vqa/blip_val_annots_okvqa_gcaption_albl_raw1017.csv.zip"

val_annotations_df = pd.read_csv(val_annotations_path)

val_annotations_df.answers = val_annotations_df.answers.apply(literal_eval)

val_annotations_df.gcaption = val_annotations_df.gcaption.apply(literal_eval)

val_annotations_df.albl = val_annotations_df.albl.apply(literal_eval)

re={}
re_img=[]
acc = []
for i in tqdm(range(val_annotations_df.shape[0])):
    test_sample = val_annotations_df.iloc[i]
    ans = test_sample.answers
    most_common_answer = max(set(ans), key=ans.count)
    ques=test_sample.question
    id = test_sample.question + str(test_sample.image_id)

    # 测试图片路径
    image_f='/home/lh/mukea-clip/data/val2014/'+test_sample.image_path

    image = Image.open(image_f)

    prompt = "<|image|><|begin_of_text|>Original Question:"+ques+"\nTo address the original question step-by-step, please provide a chain-of-thought with only three relevant questions about the original question. Three relevant questions are:"


    inputs = processor(image, prompt, return_tensors="pt").to(model.device)

    prompt_tokens = inputs.input_ids.shape[1]

    output = model.generate(**inputs, max_new_tokens=prompt_tokens+50)

    outputs=processor.decode(output[0]).split("Three relevant questions are:")[-1].split("Answer")[0].split("Please")[0].strip()

    print(outputs)
    re[id] = outputs

path0 = "train_1218_okvqa_subques.pkl"
output = open(path0, 'wb')
pickle.dump(re, output)
output.close()