from collections import defaultdict
import io
import os
import os.path as op
import numpy as np
from openai import OpenAI
import json
import base64
from tqdm import tqdm
import imageio
import pdb
import argparse
from PIL import Image
import matplotlib.pyplot as plt


DEFAULT_NUM_FRAMES = 16.0


def save_json_file(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)


# Function to encode the image
def encode_image_from_path(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image(image):
    if isinstance(image, Image.Image):  # Check if input is a PIL Image
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # Save PIL Image to buffer in JPEG format
    elif isinstance(image, np.ndarray):  # Check if input is a NumPy array
        image = Image.fromarray(image.astype("uint8"))  # Convert NumPy array to PIL Image
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # Save PIL Image to buffer in JPEG format
    else:
        raise TypeError("Input must be a PIL Image or a NumPy array")

    # Encode the buffer to base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def plot_frames(frames, indices, plot_indices, fname="tmp.jpg"):
    if len(indices) == 1:
        Image.fromarray(frames[indices[0]]).save(fname)
        return frames[indices[0]]
    h, w = frames[0].shape[:2]
    fig, axes = plt.subplots(1, len(indices), figsize=((w/h*3)*len(indices), 3))
    for i, (j, k) in enumerate(zip(indices, plot_indices)):
        axes[i].imshow(frames[j])
        axes[i].axis('off')
        axes[i].set_title(f"Frame {k}", fontsize=20)
    
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    pil_img = Image.open(fname)
    return pil_img


def plot_single_frame(frame, plot_idx, fname="tmp.jpg"):
    w = int(frame.shape[1] / frame.shape[0] * 3.0)
    fig = plt.figure(figsize=(w, 3.5))
    plt.imshow(frame)
    plt.axis("off")
    plt.title(f"Frame {plot_idx}", fontsize=20)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    pil_img = Image.open(fname)
    return pil_img


def form_gpt_question(q, frames: str, frame_idx: list, plot_frame_idx: list, single_image=True):
    if single_image:
        vision_content = [
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(plot_frames(frames, frame_idx, plot_frame_idx))}",
                },
            }
        ]
    else:
        vision_content = [
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(plot_single_frame(frames[idx], plot_idx))}",
                },
            }
            for idx, plot_idx in zip(frame_idx, plot_frame_idx)
        ]

    vision_description = ", ".join([str(idx) for idx in plot_frame_idx])
    vision_description = f"The images show Frame {vision_description} of a video. "
    text_content = [{"type": "text", "text": vision_description + q + " Answer yes or no and then justify your answer in one sentence.\n"}]
    return text_content + vision_content


def form_question(model, q, frames, frame_idx, plot_frame_idx, single_image=True):
    if model == 'gpt':
        return form_gpt_question(q, frames, frame_idx, plot_frame_idx, single_image)
    else:
        raise ValueError("Invalid model")
    

def eval_forward(model, client, inputs):
    if model == 'gpt':
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a powerful and concise vision language model that answer questions based on the images shown to you."},
                {"role": "user", "content": inputs}
                ],
            max_tokens=1000,
        )
        return response
    else:
        raise ValueError("Invalid model")


def idx_mapper(indices, n_frames):
    return sorted(set([round((idx-1)/(DEFAULT_NUM_FRAMES-1) * n_frames) for idx in indices]))


def map_score(score):
    if score < 0.9:
        return 0
    elif score > 0.98:
        return 1
    else:
        # Mapping the value linearly between 0 and 1 for the range [0.9, 0.98]
        return (score - 0.9) / (0.98 - 0.9)
    

def get_eval_scores(eval_answers, model, model_type, use_frame_consistency=False):
    prompts = json.load(open(f"./{model_type}_prompts.json"))
    id2type = {d['id']: d['type'] for d in prompts}

    avg_scores = []
    TCR = []
    TCR_by_type = defaultdict(list)
    score_by_type = defaultdict(list)
    for k, v in eval_answers.items():
        if use_frame_consistency:
            try:
                transition_completed = all([q['correct'] for q in v if q['type'] != 'other objects'])
                frame_consistency_score = json.load(open(f"eval_results/i2v_consistency/consecutive_frame_sim/i2v/{model}.json"))[k]
                score = np.mean([q['correct'] for q in v if q['type']!='transition object consistency'])
                score = 2/3*score + 1/3*map_score(frame_consistency_score)
            except Exception as e:
                transition_completed = all([q['correct'] for q in v if q['type'] != 'other objects'])
                score = np.mean([q['correct'] for q in v])    
        else:
            transition_completed = all([q['correct'] for q in v if q['type'] != 'other objects'])
            score = np.mean([q['correct'] for q in v])

        avg_scores.append(score)
        TCR.append(transition_completed)
        TCR_by_type[id2type[int(k.split("-")[0])]].append(transition_completed)
        score_by_type[id2type[int(k.split("-")[0])]].append(score)
    return TCR, TCR_by_type, np.mean(avg_scores), score_by_type


def eval_model(args):
    modelname = args.model
    meta_folder = f"{args.eval_model}_eval"
    save_dir = op.join(f"eval_results/{meta_folder}/{args.model_type}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(op.join(save_dir, "tmp"), exist_ok=True)
    os.makedirs(f"eval_results/{meta_folder}/errors", exist_ok=True)

    if args.eval_model == 'gpt':
        client = OpenAI(api_key=args.api_key)
    else:
        raise ValueError("Invalid eval model")

    eval_questions = json.load(open(f"{args.model_type}_assertions.json"))
    if not op.exists(op.join(save_dir, f"{modelname}.json")):
        answers = {} 
        errors = {}
        price = 0
        n_samples = 5 if modelname not in ['gt', 'lvd'] else 1
        for i in tqdm(range(len(eval_questions))):
            for j in range(n_samples):
                folder = f"{i:05d}-{j:05d}"

                video_file = f"generated_videos/{args.model_type}/{modelname}/{folder}.mp4"
                if modelname == "gt":
                    video_file = f"youtube_videos/{i:05d}.mp4"

                video_reader = imageio.get_reader(video_file)
                frames = [f for f in video_reader]
                total_frames = len(frames)
                sample_answers = []
                for question_meta in eval_questions[str(i)]:
                    frame_idx = idx_mapper(question_meta['frames'], total_frames-1)

                    q2gpt = form_question(args.eval_model, question_meta['question'], frames, frame_idx, question_meta['frames'], single_image=True) # single image gives better results
                    try:
                        response = eval_forward(args.eval_model, client, q2gpt)
                        gpt_answer = response.choices[0].message.content.strip().strip(".").lower()
                        price += response.usage.prompt_tokens*10/1000000 + response.usage.completion_tokens*30/1000000
                    except Exception as e:
                        errors[folder] = str(e)
                        break

                    if 'yes' in gpt_answer:
                        correct = 1
                    elif 'no' in gpt_answer:
                        correct = 0
                    else:
                        correct = 0.5

                    sample_answers.append({
                        "question": question_meta['question'],
                        "frames": question_meta['frames'],
                        "type": question_meta['type'],
                        "correct": correct,
                        "gpt_answer": gpt_answer
                    })

                if folder not in errors:
                    answers[folder] = sample_answers
                    save_json_file(answers, op.join(save_dir, f"{modelname}.json"))
                else:
                    save_json_file(errors, f"eval_results/{meta_folder}/errors/{args.model_type}_{modelname}_errors.json")
            print("Price: ", price)    
    else:
        answers = json.load(open(op.join(save_dir, f"{modelname}.json")))
    
    # TODO: add consecutive frame similarity for I2V models

    TCR, TCR_by_type, avg_score, score_by_type = get_eval_scores(answers, modelname, args.model_type, use_frame_consistency=args.model_type=='i2v')
    print(modelname)
    for k, v in TCR_by_type.items():
        print("{}: {:.3f}, {:.4f}".format(k, np.mean(v)*100, np.mean(score_by_type[k])))
    print("overall: {:.3f}, {:.4f}".format(np.mean(TCR)*100, avg_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("GPT-4V eval")
    parser.add_argument("--model_type", type=str, choices=["t2v", "i2v"])
    parser.add_argument("--model", type=str, help="This should be the model name under ./generated_videos/{i2v, t2v}/") # choices=["videocrafter", "show-1", "lavie", "modelscope", "free-bloom", "seine", "dynamicrafter", "free-bloom-videocrafter", "gt", "lvd"]
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--eval_model", type=str, choices=['gpt'], default='gpt')
    args = parser.parse_args()
    eval_model(args)