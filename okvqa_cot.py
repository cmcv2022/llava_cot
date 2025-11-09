import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
import numpy as np
import pandas as pd
from torch.nn.functional import cosine_similarity
import torch,pickle,json
from typing import List
from ast import literal_eval
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import AutoProcessor, BlipForImageTextRetrieval
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
import regex,string,random
import nltk,clip
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.data.path.append('/home/lh/nltk_data')
#
# # 初始化 lemmatizer
lemmatizer = WordNetLemmatizer()
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('wordnet')
#
# 获取单词的词性
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# 词形还原函数
def lemmatize_word(word):
    pos = get_wordnet_pos(word)
    return lemmatizer.lemmatize(word, pos)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class StopAtSpecificTokenCriteria(StoppingCriteria):
    """
    当生成出第一个指定token时，立即停止生成
    ---------------
    ver: 2023-08-02
    by: changhongyu
    """

    def __init__(self, token_id_list: List[int] = None):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # return np.argmax(scores[-1].detach().cpu().numpy()) in self.token_id_list
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list

stopping_criteria = StoppingCriteriaList()
stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[29889]))

setup_seed(888)
llama_model = LlamaForCausalLM.from_pretrained('/home/data/llama2', torch_dtype=torch.float16,device_map="auto")
llama_tokenizer = AutoTokenizer.from_pretrained('/home/data/llama2')

blip_model = BlipForImageTextRetrieval.from_pretrained("blip-itm-base-coco").to("cuda")
blip_processor = AutoProcessor.from_pretrained("blip-itm-base-coco")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# blip_model, blip_processor = clip.load("ViT-B/32",device=device)


n_shots = 1
k_ensemble = 1
MAX_CAPTION_LEN = 30
NO_OF_CAPTIONS_AS_CONTEXT = 9

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def get_context_examples(sample_q_embed, sample_i_embed, train_q_embedds, train_i_embedds, n_shots):
    """
    Get the n context examples for n-shot in context learning
    according to the avg img and question similarities

    :param sample_q_embed: The normalized question embedding of the test sample (shot)
    :param sample_i_embed: The normalized image embedding of the test sample (shot)
    :param train_q_embedds: Dataframe containing the normalized question embeddings of the train samples (shots)
    :param train_i_embedds: Dataframe containing the normalized image embeddings of the train samples (shots)
    :param n_shots: The number of training examples (shots) to return
    :returns results_df: Dataframe containing the n_shot most similar examples to the test sample


    """
    # compute question sims
    q_sims_df = train_q_embedds.copy(deep=True)
    q_sims_df['q_cos_sim'] = q_sims_df.question_embedd.apply(lambda x: np.matmul(x, sample_i_embed))

    # compute image sims
    i_sims_df = train_i_embedds.copy(deep=True)
    i_sims_df['i_cos_sim'] = i_sims_df.image_embedd.apply(lambda x: np.matmul(x, sample_i_embed))

    q0_sims_df = train_q_embedds.copy(deep=True)
    q0_sims_df['q0_cos_sim'] = q0_sims_df.question_embedd.apply(lambda x: np.matmul(x, sample_q_embed))
    #
    # # compute image sims
    i0_sims_df = train_i_embedds.copy(deep=True)
    i0_sims_df['i0_cos_sim'] = i0_sims_df.image_embedd.apply(lambda x: np.matmul(x, sample_q_embed))
    #
    results_df = pd.merge(q_sims_df, i_sims_df, on='question_id')
    q0_sims_df = q0_sims_df.drop('question_embedd', axis=1)
    results_df = pd.merge(results_df, q0_sims_df, on='question_id')
    i0_sims_df = i0_sims_df.drop('image_embedd', axis=1)
    results_df = pd.merge(results_df, i0_sims_df, on='question_id')

    results_df['avg_cos_sim'] = results_df.apply(
        lambda row: (row["q_cos_sim"] * row["i_cos_sim"] * row["q0_cos_sim"] * row["i0_cos_sim"]), axis=1)
    results_df = results_df.sort_values(by='avg_cos_sim', ascending=False)

    return results_df[:n_shots]


def sort_captions_based_on_similarity(captions, raw_image, model, processor, device="cuda", ascending=False):
    """
    Rank the qr captions based on their similarity with the image
    :param captions: The captions that will be ranked
    :param raw_image: The PIL image object
    :param model: The image-to-text similarity model (BLIP)
    :param processor: The image and text processor
    :param device: Cpu or Gpu
    :param ascending: Bool variable for ranking the captions at ascending order or not
    :returns results_df: Captions ranked
    :returns cosine_scores: The cosine score of each caption with the image
    """
    # encode the captions
    text_input = processor(text=captions, return_tensors="pt", padding=True).to("cuda")
    text_embeds = model.text_encoder(**text_input)
    text_embeds = text_embeds[0]
    text_features = model.text_proj(text_embeds[:, 0, :])

    # encode the image
    image_input = processor(images=raw_image, return_tensors="pt").to("cuda")
    vision_outputs = model.vision_model(**image_input)
    image_embeds = vision_outputs[0]
    image_feat = model.vision_proj(image_embeds[:, 0, :])

    # compute cos sim
    cosine_scores = cosine_similarity(text_features, image_feat).tolist()

    # sort captions based on the cosine scores
    captions = [x for _, x in sorted(zip(cosine_scores, captions), reverse=True)]
    cosine_scores.sort(reverse=True)
    return captions, cosine_scores

train_annotations_path="annotations/ok_vqa/blip_train_annots_okvqa_gcaption_albl_raw1017.csv.zip"
val_annotations_path="annotations/ok_vqa/blip_val_annots_okvqa_gcaption_albl_raw1017.csv.zip"
train_annotations_df = pd.read_csv(train_annotations_path)
val_annotations_df = pd.read_csv(val_annotations_path)

train_q_embedds = pd.read_csv("blip_embedds/ok_vqa/blip_normalized_q_embedds/blip_train_question_embedds.csv.zip")
train_i_embedds = pd.read_csv("blip_embedds/ok_vqa/blip_normalized_i_embedds/blip_train_image_embedds.csv.zip")
train_q_embedds.question_embedd = train_q_embedds.question_embedd.apply(literal_eval)
train_i_embedds.image_embedd = train_i_embedds.image_embedd.apply(literal_eval)

val_q_embedds = pd.read_csv("blip_embedds/ok_vqa/blip_normalized_q_embedds/blip_val_question_embedds.csv.zip")
val_i_embedds = pd.read_csv("blip_embedds/ok_vqa/blip_normalized_i_embedds/blip_val_image_embedds.csv.zip")
val_q_embedds.question_embedd = val_q_embedds.question_embedd.apply(literal_eval)
val_i_embedds.image_embedd = val_i_embedds.image_embedd.apply(literal_eval)

train_captions = pd.read_csv("question_related_captions/ok_vqa/train_data_qr_captions_csv0")
train_captions.captions = train_captions.captions.apply(literal_eval)

val_captions = pd.read_csv("question_related_captions/ok_vqa/val_data_qr_captions_csv0")
val_captions.captions = val_captions.captions.apply(literal_eval)

train_annotations_df.answers = train_annotations_df.answers.apply(literal_eval)
val_annotations_df.answers = val_annotations_df.answers.apply(literal_eval)

train_annotations_df.gcaption = train_annotations_df.gcaption.apply(literal_eval)
val_annotations_df.gcaption = val_annotations_df.gcaption.apply(literal_eval)

train_annotations_df.albl = train_annotations_df.albl.apply(literal_eval)
val_annotations_df.albl = val_annotations_df.albl.apply(literal_eval)

train_images_dir="/home/lh/mukea-clip/data/train2014/"
val_images_dir="/home/lh/mukea-clip/data/val2014/"

llama_answers = []
question_id_list, image_id_list = [], []

with open('/home/lh/revive/test_0822_okvqa_rewrite.pkl', 'rb') as f:
    ww = pickle.load(f)

with open('/home/lh/revive/processed_data/okvqa_id_val.json', 'rb') as f_six:
    ofaid=json.load(f_six)

with open('/home/lh/revive/processed_data/okvqa_val.json', 'rb') as f:
    lbl =json.load(f)

with open('/home/lh/llava_cot/val_1218_okvqa_subques.pkl', 'rb') as f:
    subques = pickle.load(f)

with open('/home/lh/llava_cot/val_1218_okvqa_subquesans.pkl', 'rb') as f:
    subquesans = pickle.load(f)

with open('/home/lh/llava_cot/val_1228_okvqa_rethink_subquesans.pkl', 'rb') as f:
    subqa = pickle.load(f)

acc=[]
for i in tqdm(range(val_annotations_df.shape[0])):

    test_sample = val_annotations_df.iloc[i]
    ans=test_sample.answers
    id = test_sample.question + str(test_sample.image_id)
    qid=ofaid[id]
    label=','.join(lbl[qid]['label'])
    know=ww[id]
    sample_q_embed = val_q_embedds[val_q_embedds.question_id == test_sample.question_id].question_embedd.iloc[0]
    sample_i_embed = val_i_embedds[val_i_embedds.question_id == test_sample.question_id].image_embedd.iloc[0]
    get_context_examples_df = get_context_examples(sample_q_embed, sample_i_embed,
                                                   train_q_embedds, train_i_embedds, n_shots=n_shots * k_ensemble)
    get_context_examples_df = pd.merge(train_annotations_df,
                                       get_context_examples_df[['question_id', 'avg_cos_sim']], on='question_id')

    # perform few shot in context learning for this test sample
    pred_answer_list, pred_prob_list = [], []
    for k in range(k_ensemble):  # we use k promts for each test sample
        prompt = 'Please answer the question according to the context. \n===\n'
        for ni in range(n_shots):
            # take the id of the n-th shot
            if get_context_examples_df is None:
                context_key = train_annotations_df.sample(1, random_state=ni)
            else:
                context_key = get_context_examples_df.iloc[ni + n_shots * k]

            try:
                raw_image = Image.open(train_images_dir + context_key.image_path)
            except:
                raw_image = Image.open(val_images_dir + context_key.image_path)

            # get captions
            context_key_captions = train_captions[train_captions.question_id == context_key.question_id].iloc[
                0].captions

            context_key_captions, cos_scores = sort_captions_based_on_similarity(context_key_captions,
                                                                                 raw_image=raw_image,
                                                                                 model=blip_model,
                                                                                 processor=blip_processor,
                                                                                 device="cuda",ascending=False)
            context_key_answers = \
            train_annotations_df[train_annotations_df.question_id == context_key.question_id].iloc[0].answers
            most_common_answer = max(set(context_key_answers),
                                     key=context_key_answers.count)  # most common answer for this context example

            prompt += 'Context:\nStart of Context:\n'
            for j, caption in enumerate(context_key_captions[:NO_OF_CAPTIONS_AS_CONTEXT]):
                caption = " ".join(caption.split()[:MAX_CAPTION_LEN])  # truncate
                if j < NO_OF_CAPTIONS_AS_CONTEXT - 1:
                    prompt += '%s,\n' % caption
                else:
                    prompt += '%s,\n%s\n%s\nEnd of Context\n' % (caption,context_key.gcaption[0],','.join(context_key.albl[:3]))
            #         prompt += '%s\nEnd of Context\n' % caption
            # prompt += '%s\n%s\nEnd of Context\n' % (context_key.gcaption[0], ','.join(context_key.albl[:10]))
            # prompt += 'Question: %s\nAnswer: %s\n\n===\n' % (context_key.question, most_common_answer)

            if ni<n_shots-1:
                prompt += 'Question: %s\nAnswer: %s\n\n===\n' % (context_key.question, most_common_answer)
            else:
                prompt += 'Question: %s\nAnswer: %s\n\n' % (context_key.question, most_common_answer)

        # sub_question = subques[id]
        # sub_questiones = subques[id][0].split('\n')
        # for kk, jj in enumerate(sub_questiones):
        #    if kk < 3:
        #        prompt += 'Question:' + jj + '\nAnswer:' + subquesans[id][kk] + '\n\n===\n'

        sub_questiones = subqa[id].split('The image')[0] + '\n\n'

        prompt += sub_questiones

        #get captions of the test sample
        test_sample_captions = val_captions[val_captions.question_id == test_sample.question_id].iloc[0].captions
        raw_test_image = Image.open(val_images_dir + test_sample.image_path)
        # test_sample_captions.append(test_sample.gcaption[0])
        # test_sample_captions.append(','.join(test_sample.albl[:10]))

        # sort the captions based on the cos sim
        test_sample_captions, cos_scores = sort_captions_based_on_similarity(test_sample_captions,
                                                                             raw_image=raw_test_image,
                                                                             model=blip_model,
                                                                             processor=blip_processor,
                                                                             device="cuda",
                                                                            ascending=False)
        prompt += 'Context:\nStart of Context:\n'
        for j, caption in enumerate(test_sample_captions[:NO_OF_CAPTIONS_AS_CONTEXT]):
            caption = " ".join(caption.split()[:MAX_CAPTION_LEN])  # truncatte
            if j < NO_OF_CAPTIONS_AS_CONTEXT - 1:
                prompt += '%s,\n' % caption
            else:
                prompt += '%s,\n%s\n%s\nEnd of Context\n' % (caption,test_sample.gcaption[0],','.join(test_sample.albl[:3]))
        #          prompt += '%s\nEnd of Context\n' % caption
        # prompt += '%s\n%s\nEnd of Context\n' % (test_sample.gcaption[0],','.join(test_sample.albl[:10]))

        prompt += 'Question: %s\nAnswer:' % test_sample.question



        # prompt = "Hey, are you conscious? Can you talk to me?"
        inputs = llama_tokenizer(prompt, return_tensors="pt")

        # Generate
        prompt_tokens = inputs.input_ids.shape[1]
        input_ids = inputs.input_ids.to("cuda")
        outputs = llama_model.generate(input_ids, max_length=prompt_tokens + 5, num_beams=2,
                                       return_dict_in_generate=True, output_scores=True, num_return_sequences=1,
                                       do_sample=False, temperature=1.0, top_p=1.0)
        outputs_sequences, outputs_sequences_scores = outputs.sequences, outputs.sequences_scores
        pred_answer = llama_tokenizer.batch_decode(outputs_sequences[:, prompt_tokens:], skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False)[0].split('\n')[0]
        # print(pred_answer.replace("=", "").strip())
        # print(np.exp(outputs_sequences_scores.item()))
        # llama_preds_df['llama_answer'] = llama_preds_df['llama_answer'].apply(lambda x: x.replace("=", "").strip())
        pred_answer_list.append(pred_answer.replace("=", "").strip())
        pred_prob_list.append(np.exp(outputs_sequences_scores.item()))

    # # take the sequence with the max score
    max_prob = max(pred_prob_list)
    max_index = pred_prob_list.index(max_prob)
    ansp=pred_answer_list[max_index]
    print(test_sample.question)
    print(ans)
    print(ansp)
    acc.append(ems(ansp, ans))
    print(np.mean(acc))
    torch.cuda.empty_cache()
print(np.mean(acc))



