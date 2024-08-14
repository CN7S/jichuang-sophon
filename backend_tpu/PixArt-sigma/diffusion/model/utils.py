
def prepare_prompt_ar(prompt):
    prompt_clean = prompt.split("--aspect_ratio")[0].split("--ar")[0].split("--hw")[0]
    return prompt_clean

