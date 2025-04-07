import os
import glob
import re


def iterate_repository_file(repo_dir, suffix):
    pattern = os.path.join(repo_dir, "**", f"*.{suffix}")
    files = glob.glob(pattern, recursive=True)
    return files


def extract_code(context_str):

    pattern = f'```[a-z_A-Z]*\n(.*?)\n```'
    matches = re.findall(pattern, context_str, re.DOTALL)
    if len(matches) == 0:
        return ''
    else:
        return matches[0]


def extract_code_from_response(context_str: str):
    context_str = context_str.removesuffix("END_OF_CODE")
    code = extract_code(context_str)
    if len(code) == 0:
        code = context_str
    return code.removesuffix("END_OF_CODE")
