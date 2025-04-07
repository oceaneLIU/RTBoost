import openai
import argparse
import anthropic
import pandas as pd
from tqdm import tqdm
from retriever import Retriever
from metrics import cal_metrics
from graph_aligner import GraphAligner
from rdfs.dependency_graph.models.language import Language
from incremental_task_builder import IncrementalTaskBuilder


def parser_args():
    parser = argparse.ArgumentParser(description="Generate response from llm")
    parser.add_argument('--model', default='gpt-4o-2024-11-20', type=str)
    parser.add_argument('--source_language', type=str)
    parser.add_argument('--target_language', type=str)
    parser.add_argument('--alpha', type=float, default=0.6)
    return parser.parse_args()


def main(args):

    if "gpt" in args.model:
        model = openai.OpenAI(base_url="xxxxxx", api_key="xxxxxx")
    elif "claude" in args.model:
        model = anthropic.Anthropic()
    elif "deepseek" in args.model:
        model = openai.OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")
    print('Model loading finished')

    if args.source_language == "arkts" or args.target_language == "arkts":
        df = pd.read_excel(f"./RepoTrans/java_arkts/java_arkts.xlsx")
        repo_dir = "RepoTrans/java_arkts/"
        repo_language1 = Language.Java
        repo_language2 = Language.ArkTS
    else:
        df = pd.read_excel(f"./RepoTrans/java_csharp/java_csharp.xlsx")
        repo_dir = "RepoTrans/java_csharp/"
        repo_language1 = Language.Java
        repo_language2 = Language.CSharp

    task_dict = dict()
    for i in range(0, len(df)):
        repo1_name = "/".join(df.loc[i][f'{args.source_language}_path'].split('/')[0:2])
        repo2_name = "/".join(df.loc[i][f'{args.target_language}_path'].split('/')[0:2])
        repo_name = (repo1_name, repo2_name)
        if repo_name not in task_dict.keys():
            task_dict[repo_name] = dict()
        task_dict[repo_name][i] = list(df.loc[i])
    prompt_col_name = f"{args.source_language}2{args.target_language}_prompt"
    for repo_name, repo_trans_task in task_dict.items():
        print(f"Repo_name: {repo_name}")
        repo1_path = repo_dir + repo1_name
        repo2_path = repo_dir + repo2_name
        incremental_task_builder = IncrementalTaskBuilder(repo_name[0], repo1_path, repo_language1,
                                                          repo_name[1], repo2_path, repo_language2,
                                                          repo_trans_task,
                                                          args.source_language, args.target_language)
        # for each task
        for i in tqdm(range(0, len(incremental_task_builder.task_sorted_list))):

            task_id = incremental_task_builder.task_sorted_list[i]
            graph1, graph2 = incremental_task_builder.construct_incremental_translation_task(i, args.target_language)
            graph_aligner = GraphAligner(graph1.graph, graph2.graph,
                                         repo_language1, repo_language2,
                                         similarity="bag-of-words",
                                         alpha=args.alpha)
            alignment = graph_aligner.get_layerwise_alignment()
            retriever = Retriever(graph1, graph2, repo_language1, repo_language2, alignment, similarity="bag-of-words")
            prompt, code = retriever.retrival_prompt(repo_dir + df.iloc[task_id][f'{args.source_language}_path'],
                                                       df.iloc[task_id][f'{args.source_language}_start_line'],
                                                       df.iloc[task_id][f'{args.source_language}_end_line'],
                                                       df.iloc[task_id][f'{args.source_language}_method'],
                                                       args.source_language, args.target_language)
            df.loc[int(task_id), f"{args.source_language}2{args.target_language}_prompt"] = prompt
            if "gpt" in args.model or "deepseek" in args.model:
                completion = model.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    top_p=0,
                    n=1,
                    max_tokens=512
                )
                response = completion.choices[0].message.content
            elif "claude" in args.model:
                completion = model.messages.create(
                    model=args.model,
                    max_tokens=512,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    top_p=0,
                    n=1,
                )
                response = completion.content[0].text
            df.loc[int(i), f"{prompt_col_name}"] = prompt
            df.loc[int(i), f"{prompt_col_name}_{args.model}"] = response
            df.to_excel(f"./results/{args.source_language}2{args.target_language}_{args.model}.xlsx", index=False)


if __name__ == "__main__":
    args = parser_args()
    main(args)
    excel_name = f"./results/{args.source_language}2{args.target_language}_{args.mode}_{args.model}.xlsx"
    prediction_col_name = f"{args.source_language}2{args.target_language}_prompt_{args.model}"
    cal_metrics(excel_name, prediction_col_name, args.target_language)
