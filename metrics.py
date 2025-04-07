import pandas as pd
from utils import extract_code_from_response
from bleu.bleu import compute_bleu
from bleu.CodeBLEU.calc_code_bleu import make_weights
from bleu.CodeBLEU import bleu, weighted_ngram_match, syntax_match, dataflow_match


def code_string_preprocessing(code_str: str):
    code_lines = [line.strip() for line in code_str.splitlines() if line.strip()]
    code_lines = [line for line in code_lines if not line.startswith('//')]
    return "".join(code_lines)


def exact_match(ground_truth: str, prediction: str):
    target_lines_str = code_string_preprocessing(ground_truth)
    prediction_lines_str = code_string_preprocessing(prediction)
    return int(target_lines_str == prediction_lines_str)


def bleu_score(ground_truth: str, prediction: str):
    max_order = 4
    smooth = True
    gt_str = code_string_preprocessing(ground_truth)
    pred_str = code_string_preprocessing(prediction)

    translations = [pred_str.strip().split()]
    reference = [[gt_str.strip().split()]]

    bleu_score, _, _, _, _, _ = compute_bleu(reference, translations, max_order, smooth)
    return bleu_score


def codebleu_score(ground_truth: str, prediction: str, lang: str):
    gt_str = code_string_preprocessing(ground_truth)
    pred_str = code_string_preprocessing(prediction)

    alpha, beta, gamma, theta = 0.25, 0.25, 0.25, 0.25

    pre_references = [[gt_str.strip()]]
    hypothesis = [pred_str.strip()]

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(hypothesis)

    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    keywords = [x.strip() for x in open('./bleu/CodeBLEU/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'. \
          format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    code_bleu_score = alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score

    return code_bleu_score


def cal_metrics(excel_name, prediction_col_name, target_language):
    df = pd.read_excel(excel_name)
    df = df.fillna('')

    gt_col_name = f"{target_language}_method"
    all_em, all_bleu, all_codebleu = 0, 0, 0
    for i in range(0, len(df)):
        gt_code = df.loc[i][gt_col_name]
        pred_str = df.loc[i][prediction_col_name]
        pred_code = extract_code_from_response(pred_str)
        em = exact_match(gt_code, pred_code)
        all_em += em
        df.loc[i, "em"] = em

        bleu_val = bleu_score(gt_code, pred_code)
        all_bleu += bleu_val
        df.loc[i, "bleu"] = bleu_val

        codebleu = codebleu_score(gt_code, pred_code, target_language)
        all_codebleu += codebleu
        df.loc[i, "codebleu"] = codebleu
        print(em, bleu_val, codebleu)

    print(f"Average EM: {all_em / len(df)}")
    print(f"Average BLEU: {all_bleu / len(df)}")
    print(f"Average CodeBLEU: {all_codebleu / len(df)}")
    df.to_excel(excel_name, index=False)
    return


def alignment_accuracy(groundtruth_alignment: dict, predicted_alignment: list):
    correct = 0
    wrong = dict()
    visited = set()
    for v, u in predicted_alignment:
        v_str = str(v.location.file_path) + "@:" + str(v.location.end_line)
        u_str = str(u.location.file_path) + "@:" + str(u.location.end_line)
        if v_str in groundtruth_alignment.keys():
            if u_str == groundtruth_alignment[v_str] and v_str not in visited:
                correct += 1
            else:
                t = v.type
                if t not in wrong.keys():
                    wrong[t] = 1
                else:
                    wrong[t] += 1
            visited.add(v_str)
    return correct / len(groundtruth_alignment), correct



