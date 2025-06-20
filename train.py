import ast 
from tqdm import tqdm
from transformers import AutoTokenizer , AutoModelForCausalLM , BitsAndBytesConfig
from peft import get_peft_model , LoraConfig , prepare_model_for_kbit_training
import bitsandbytes as bnb
import pandas as pd 
import transformers
from datasets import Dataset
import torch 
from trl import SFTTrainer

def format_path(path) : 

    path = path.replace('\n' , '')
    path = path.replace('\t' , '')
    path = path.replace(' ' , '')

    return path


with open(format_path(
    '''
    question_answer_pairs.jsonl
    '''
)) as fil : question_answer_pairs = [
    ast.literal_eval(row)
    for row 
    in fil
]

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

    Passage : 
'''

processed = []

for pair in tqdm(question_answer_pairs , total = len(question_answer_pairs)) : 
    
    pair = pair['text']
    pair = pair.replace(removal_prompt , '')
    
    for charac in pair : 
        
        if charac == ' ' or charac == '\n' : pair = pair[1 :]
        else : break
            
    processed.append(pair)

tokenizer = AutoTokenizer.from_pretrained(
    format_path(
        '''
        meta-llama/
            Llama-2-7b-hf
        '''
    ) , 
    padding = True , 
    truncation = True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    format_path(
        '''
        meta-llama/
            Llama-2-7b-hf
        '''
    ) ,
    device_map = {'' : 0} ,
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True ,
        bnb_4bit_compute_dtype = torch.float16 ,
        bnb_4bit_quant_type = 'nf4' ,
        bnb_4bit_use_double_quant = False 
    )
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

cls = bnb.nn.Linear4bit
modules = set()

for name , module in model.named_modules() :

    if isinstance(module , cls) :

        names = name.split('.')
        modules.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in modules : modules.remove('lm_head')

lora_config = LoraConfig(
    r = 64 ,
    lora_alpha = 32 ,
    target_modules = modules ,
    lora_dropout = 0.05 ,
    bias = 'none' ,
    task_type = 'SEQ_2_SEQ'
)

model = get_peft_model(model, lora_config)

args = transformers.TrainingArguments(
    max_steps = 25 ,
    logging_steps = 1 ,
    warmup_steps = 0.03 ,
    learning_rate = 2e-4 ,
    output_dir = 'outputs' ,
    save_strategy = 'epoch' ,
    optim = 'paged_adamw_8bit' ,
    per_device_train_batch_size = 1 ,
    gradient_accumulation_steps = 4
)

collator = transformers.DataCollatorForLanguageModeling(
    tokenizer , 
    mlm = False
)

data = pd.DataFrame({
    'text' : processed
})

data = Dataset.from_pandas(data)

trainer = SFTTrainer(
    args = args ,
    model = model ,
    data_collator = collator ,
    peft_config = lora_config ,
    train_dataset = data ,
    dataset_text_field = 'text' , 
    max_seq_length = 500
)

trainer.train()
trainer.model.save_pretrained('llama2_instructed')