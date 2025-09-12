from string import Template


def get_prompt_template_for_summarizing_text():
    
    prompt_tmp = '''
You will be given a piece of text. Your task is to \
summarize the text in a concise and clear manner, \
capturing the main ideas and key points while maintaining the original meaning.

### Text to Summarize

$text

### Instructions

- Provide a summary that is brief yet comprehensive.
- Ensure that the summary accurately reflects the content of the original text.
- Avoid adding any personal opinions or interpretations.
- Do not output anything other than the summary.

Begin your response below:
    '''.strip()

    return Template(prompt_tmp)


def get_prompt_template_for_writing_setting1():

    prompt_tmp = '''
You will be given one or more writing samples from a specific author. \
Your task is to analyze the author's style, tone, and voice, then craft a new piece of $genre \
that closely mimics their writing based on a provided summary. Your writing should be around $num_words words.

### Author's Writing Sample(s)

$writing_samples

### Writing Task Summary

$summary

### Instructions

- Ensure your writing faithfully replicates the author's style, including tone, word choices, and sentence structure, etc.
- Maintain consistency with the author's voice while accurately reflecting the details of the given summary.
- Strive to make your writing indistinguishable from the original author's work.
- Do not output anything other than the writing.

Begin your response below:
'''.strip()

    return Template(prompt_tmp)


def get_prompt_template_for_writing_setting2():
    return get_prompt_template_for_writing_setting1()



def get_prompt_template_for_writing_setting3():
    return get_prompt_template_for_writing_setting1()


def get_prompt_template_for_writing_setting4():
    prompt_tmp = '''
Given the following summary, your task is to generate a \
writing sample around $num_words words. The genre of the \
writing is $genre. Do not output anything other than the writing.

### Writing Task Summary

$summary

Begin your response below:
'''.strip()

    return Template(prompt_tmp)


def get_prompt_template_for_writing_setting5():

    summary_only_prompt_tmp = '''
Given the following summary, your task is to generate a continuation for \
the provided human-authored text snippet with around $num_words words. \
The genre of the writing is $genre. Do not output anything other than the writing. 

### Writing Task Summary

$summary

### Human-Authored Text Snippet

$snippet

Begin your response below:
'''.strip()
    
    exemplars_plus_summary_prompt_tmp = '''
You will be given one or more writing samples from a specific author plus \
a text snippet of $genre from the same author. Your task is to analyze the author's style, \
tone, and voice, then generate a continuation for the provided human-authored text snippet \
with around $num_words words that closely mimics their writing based on a provided summary. 

### Author's Writing Sample(s)

$writing_samples

### Writing Task Summary

$summary

### Human-Authored Text Snippet

$snippet

### Instructions

- Ensure your writing faithfully replicates the author's style, including tone, word choices, and sentence structure, etc.
- Maintain consistency with the author's voice while accurately reflecting the details of the given summary.
- Strive to make your writing indistinguishable from the original author's work.
- Do not output anything other than the writing.

Begin your response below:
'''.strip()

    return [Template(summary_only_prompt_tmp), 
            Template(exemplars_plus_summary_prompt_tmp)]