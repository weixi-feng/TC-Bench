from collections import defaultdict
import os
import pdb
import re
from openai import OpenAI
import json
from tqdm import tqdm
import argparse


prompt1 = "A chameleon changing from brown to bright green."
sample1 = """
Transition object: chameleon, start: brown, end: bright green
other objects: None
- Check "Transition Completion"
Input: Frame 1
Q: Is there a brown chameleon in the image?
Input: Frame 1, 16
Q: Compared to Frame 1, is the chameleon bright green in Frame 16?
Input: Frame 1, 9
Q: Is the chameleon's color in Frame 9 different from its color in Frame 1, showing a sign of transition?
Input: Frame 1, 5, 9, 13, 16
Q: Do these frames show a consistent transition of chameleon changing color from brown to bright green?
- Check "Transition object consistency"
Input: Frame 1, 6
Q: Aside from color difference, do these frames show the same chameleon?
Input: Frame 1, 11
Q: Aside from color difference, do these frames show the same chameleon?
Input: Frame 1, 16
Q: Aside from color difference, is the chameleon recognizable as the same chameleon in these frames?
- Check "Other objects"
None
"""

prompt2 = "A man passing a ball from his left hand to his right hand."
sample2 = """
Transition object: ball, start: left hand, end: right hand
other objects: man
- Check "Transition Completion"
Input: Frame 1
Q: Is there a ball on the man's left hand in the image?
Input: Frame 1, 16
Q: Compared to Frame 1, is the ball on the man's right hand in Frame 16?
Input: Frame 1, 9
Q: Compared to Frame 1, is the ball between the man's left hand and right hand in Frame 9, showing a sign of movement?
Input: Frame 1, 5, 9, 13, 16
Q: Do these frames show a ball being passed from left hand to right hand?
- Check "Transition object consistency"
Input: Frame 1, 6
Q: Aside from position difference, do these two frames show the same ball?
Input: Frame 1, 11
Q: Aside from position difference, do these two frames show the same ball?
Input: Frame 1, 16
Q: Aside from position difference, do these two frames show the same ball?
- Check "Other objects"
Input: Frame 1
Q: Is there a man with a ball in his hand in the image?
Input: Frame 1, 6, 11, 16
Q: Do all the frames show the same man?
"""


prompt3 = "A bench by a lake from foggy morning to sunny afternoon."
sample3 = """
Transition object: background, start: foggy morning, end: sunny afternoon
Other objects: bench, lake
- Check "Transition Completion" 
Input: Frame 1
Q: Is the image showing a foggy morning?
Input: Frame 1, 16
Q: Comapred to Frame 1, is Frame 16 showing a sunny afternoon?
Input: Frame 1, 9
Q: Is the weather/background in Frame 9 different from that in Frame 1, showing a sign of transition?
Input: Frame 1, 5, 9, 13, 16
Q: Do these frames show the background changing from foggy morning to sunny afternoon?
- Check "Transition object consistency"
None: background is an abstract concept without a physical form
- Check "Other objects"
Input: Frame 1
Q: Is there a bench by a lake in the image?
Input: Frame 1, 6, 11, 16
Q: Do all the frames show the same bench?
Input: Frame 1, 6, 11, 16
Q: Do all the frames show the same lake?
"""


task_instruction = "Given a video description that describes a video, generate question-frame pairs to verify important components in the description. " \
"Each prompt describes a transformation/transition of an object's attribute, or an object's position or the background. " \
"Identify the transition object, its start and end status/place, and other objects, and ask assertions to verify them. \n" \
"Below are three examples showing three different types of transitions. If the assertion mention a frame, or ask for comparison, make sure you include all necessary frame indices. " \
"If a video perfectly aligns with the prompt, the answer to all the assertion questions should be \"Yes\". \n" \
"Based on the instruction and examples, generate assertions for the given prompt."


# Define a function to extract the information
def extract_checks(text):
    # Split the text into parts by "- Check"
    parts = text.split("- Check ")
    checks = []
    
    # Regular expression for finding frame numbers
    frame_regex = re.compile(r'Frame (\d+(?:, \d+)*)')
    assertion_regex = re.compile(r'Q: (.+)')
    
    # Iterate over parts and extract relevant information
    for part in parts[1:]:  # Skip the first item which is not a check
        # Get the check type by taking the first line
        check_type = part.splitlines()[0].strip('" ')
        # Find all the frames
        frames = frame_regex.findall("\n".join([l for l in part.split("\n") if l.startswith("Input:")]))
        # Convert frame strings to lists of integers
        frame_lists = [list(map(int, frame_group.split(', '))) for frame_group in frames]
        # Find all the assertions
        assertions = assertion_regex.findall(part)
        # Combine frames with their corresponding assertions
        assert len(frame_lists) == len(assertions)
        for frames, assertion in zip(frame_lists, assertions):
            checks.append({
                "frames": sorted(set([min(idx, 16) for idx in frames])), 
                "assertion": assertion,
                "type": check_type.lower()})
        
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to generate questions using OpenAI API')
    parser.add_argument('--file', type=str, required=True, help='File containing the prompts')
    parser.add_argument('--api_key', type=str, required=True, help='API key for OpenAI')
    args = parser.parse_args()

    api_key = args.api_key
    save_file = os.path.basename(args.file).replace(".json", "_assertions.json")

    data = json.load(open(args.file))
    client = OpenAI(api_key=api_key)

    attribute_binding_prompt = [
        {"role": "system", "content": task_instruction},
        {"role": "user", "content": f"Description: {prompt1}\n"},
        {"role": "assistant", "content": sample1},
        {"role": "user", "content": f"Description: {prompt2}\n"},
        {"role": "assistant", "content": sample2},
        {"role": "user", "content": f"Description: {prompt3}\n"},
        {"role": "assistant", "content": sample3},
    ]

    try:
        answers = json.load(open(save_file))
    except:
        answers = {}
    
    price = 0
    for d in tqdm(data):
        if d['id'] in answers: continue

        response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=attribute_binding_prompt+[
            {"role": "user", "content": f"Description: {d['prompt']}\n"},
        ]
        )
        answer = response.choices[0].message.content        
        try:
            check_list = extract_checks(answer)
            assert len([l for l in answer.split("\n") if l.startswith("Q:")]) == len(check_list)
            answers[d['id']] = check_list
        except Exception as e:
            print(e)
            pdb.set_trace()
        
        json.dump(answers, open(save_file, "w"), indent=4, sort_keys=True)

        price += response.usage.completion_tokens*60/1000000+response.usage.prompt_tokens*30/1000000
        print(price)