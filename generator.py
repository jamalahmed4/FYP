import torch 
from transformers import AutoTokenizer , AutoModelForCausalLM
from tqdm import tqdm
import json


prompt = '''<s>

<|user|>
Instruct :
* You will be given a passage of text as input.
* Your task is to generate one question-answer pair based on the information in the passage.
* Only ask direct subject question, not true and False or Multiple Choice
* Focus on factual questions that can be directly answered from the provided text.
* Keep the answers accurate and informative, directly addressing the corresponding question.
* The question should not be based on names, emails etc
* Do not ask vague questions like What is the context of provided text
* Only return the question answer pair and nothing else

Passage : {}

<|end|>
<|assistant|>
'''

removal_prompt = '''<s> 

<|user|> Instruct :
* You will be given a passage of text as input.
* Your task is to generate one question-answer pair based on the information in the passage.
* Only ask direct subject question, not true and False or Multiple Choice
* Focus on factual questions that can be directly answered from the provided text.
* Keep the answers accurate and informative, directly addressing the corresponding question.
* The question should not be based on names, emails etc
* Do not ask vague questions like What is the context of provided text
* Only return the question answer pair and nothing else

Passage :'''# Do not change

start_inst = '<|end|><|assistant|>'
end_inst = '<|end|>'

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_question_answer_pairs(text , model_chunk_length = 500 , max_chunk_input_length = 1024 , max_chunk_output_length = 1024) : 

    chunks = [
        text[index : index + model_chunk_length]
        for index
        in range(0 , len(text) , model_chunk_length)
    ][: 7000]

    questions = []

    for chunk in tqdm(chunks , total = len(chunks) , desc = 'Getting Question Answer Pairs') :

        text = prompt.format(chunk)

        inputs = tokenizer(
            text ,
            return_tensors = 'pt' ,
            return_attention_mask = True
        )

        if inputs['input_ids'].shape[1] > max_chunk_input_length : pass
        else :

            outputs = model.generate(**inputs, max_length=max_chunk_output_length)
            out = tokenizer.batch_decode(outputs)[0]

            out = out.replace(removal_prompt , '')
            out = out.replace(chunk , '')
            out = out.replace(start_inst , '')
            out = out.replace(end_inst , '')
            out = out.replace('Question: ' , '')
            out = out.replace('Answer: ' , '')

            for charac in out :

                if charac == '\n' or charac == ' ' : out = out[1 :]
                else : break

            questions.append(out)

    with open('question_answer_pairs.jsonl', 'w') as f :

        for question in questions:

            json_dict = {'text': question}
            f.write(json.dumps(json_dict) + '\n')
            
    return questions