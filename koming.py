import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

tokenizer = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)

model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype='auto', low_cpu_mem_usage=True
).to(device='cuda', non_blocking=True)

_ = model.eval()

context = """아래는 친구같은 척척박사 로봇 코밍과 학생의 대화입니다. 역사, 화학과 관련한 질문에 친절하게 답해줍니다.

학생: 동학농민운동이 뭐야?
코밍: 1894년에 일어난 동학농민운동은 반봉건, 반외세의 기치 아래 일어난 농민운동이야. 전봉준, 손화중, 김개남 등이 주도했어.
"""

def generator(prompt, origin_prompt):
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
    gen_tokens = model.generate(
        tokens,
        temperature=0.8,
        max_length=len(tokens[0]) + 4,
        num_beams=4
    )
    
    generated = tokenizer.batch_decode(gen_tokens)[0]
    if(len(generated[(len(origin_prompt) + 1):]) > 72):
        return generated + "..."
    
    if generated[(len(origin_prompt) + 1):].find("학생: ") != -1:
        return generated
    
    return generator(generated, origin_prompt)

logs = []

def getAnswer(question):
    gc.collect()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        logs.append("학생: " + question)
        inputs = context + "\n".join(logs[-2:]) + "\n코밍: "
        generated = generator(inputs, inputs)

        say = generated[len(inputs):].split("학생: ")[0].strip().replace("코밍:", "")
        logs.append("코밍: " + say)
        return say
