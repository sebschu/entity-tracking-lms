import os
import json
import re
import argparse

import pandas as pd


MODEL_NAME = "base"

_MODIFIERS = ["big", "small", "blue", "green", "red", "yellow"]
_MODIFIERS_REGEX_STR = "(" + "|".join(_MODIFIERS) + ")"

BOLD_START = "\033[1m"
BOLD_END = "\033[0;0m"


def parse_intial_states(first_context, zero_shot=False):
    """
      Parses the state from an initial description.
    Args:
        first_context (_type_): The initial description.
        zero_shot (bool, optional): Whether to use the zero-shot/few-shot format. Defaults to False.

    Returns:
        _type_: A dictionary with the box name as the key and the contents as value.
    """
    parts = first_context.split(", ")
    states = {}
    if zero_shot:
        for _, part in enumerate(parts):
            tokens = part.split(" ")
            if tokens[-1] == "nothing":
                states[" ".join(tokens[:-2])] = "nothing"
            else:
                is_idx = (
                    tokens.index("has") if "has" in tokens else tokens.index(
                        "contains")
                )
                states[" ".join(tokens[:is_idx])] = " ".join(tokens[is_idx+1:])
    else:
        for _, part in enumerate(parts):
            tokens = part.split(" ")
            # alt forms
            if tokens[0].lower() == "the" or tokens[0].lower() == "there":
                if "nothing" in tokens:
                    states[" ".join(tokens[-2:])] = "is empty"
                else:
                    # The {contents} is/are in Box X
                    states[" ".join(tokens[-2:])] = "contains " + \
                        " ".join(tokens[:-4]).replace("The ", "the ")
            else:
                if tokens[-1] == "empty":
                    # Box X is empty
                    states[" ".join(tokens[:-2])] = " ".join(tokens[-2:])
                else:
                    # Box X contains {content}
                    is_idx = (
                        tokens.index("has") if "has" in tokens else tokens.index(
                            "contains")
                    )
                    states[" ".join(tokens[:is_idx])] = " ".join(
                        tokens[is_idx:])

    return states


def compute_metrics(file_path, gold_file_path=None, zero_shot=False):
    """ Computes detailed metrics for each example

    Args:
        file_path (str): Path to model output (in TSV format).
        gold_file_path (str, optional): Path to gold data (in jsonl format). Defaults to None.
        zero_shot (bool, optional): Set to true if few-shot/zero-shot format is used. Defaults to False.

    Returns:
        Tuple[pandas.DataFrame,pandas.DataFrame]: Data frame with per-example and per-state info.
    """

    gold_f = None

    if gold_file_path is not None:
        gold_f = open(gold_file_path, encoding="UTF-8")

    row_state = {"correct": 1}

    total = 0
    rows = []
    rows_state = []
    with open(file_path, encoding="UTF-8") as f:
        for line in f:
            idx = ""
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                idx, gold, pred, context = parts
                if idx == "":
                    continue
            else:
                gold, pred, context = parts

            # replace "Container" w/ "Box"
            context = context.replace("Container", "Box")
            gold = gold.replace("Container", "Box")
            pred = pred.replace("Container", "Box")

            # remove zero-shot instruction
            if zero_shot:
                context = context.replace(
                    "Given the following description, complete the final sentence with the correct items that are inside the box. ",
                    "",
                )
                gold = gold.strip(".")
                pred = pred.strip(".")

            if zero_shot and "Description: " in context:
                context = context.strip("\"")
                context = context.replace("\"\"", "\"")
                context_parts = context.split("Description: ")
                context = context_parts[-1]
                context = context.replace(
                    " Statement:", "").replace("\\nStatement:", "")
                if not context.endswith(" ."):
                    context = context + " ."

            # pred = pred.lower()
            # gold = gold.lower()

            row = {}
            gold_data = None
            if gold_f is not None:
                gold_data = json.loads(gold_f.readline())
                gold_data["sentence"] = gold_data["sentence"].replace(
                    "Container", "Box")
                assert gold_data["sentence"][:-1] == (context[:-1] + gold), (
                    gold_data["sentence"][:-1] + "|||" + (context[:-1] + gold)
                )
                row["numops_local"] = gold_data["numops"]

            # attach adjectives to nouns:
            for mod in _MODIFIERS:
                gold = gold.replace(f"{mod} ", f"{mod}_")
                pred = pred.replace(f"{mod} ", f"{mod}_")

            # compute global numops
            row["numops_global"] = gold_data["sentence"][:-1].count(".") - 1
            row_state["numops_global"] = row["numops_global"]

            initial_state_required = "nothing" in gold or "is empty" in gold
            ambiguous_pragmatic_mention = False
            involves_move_content = False

            pred = pred.strip()
            gold = gold.strip()

            box_start = gold_data["sentence"][:-1].rfind("Box ") + 4
            box_no = gold_data["sentence"][box_start:box_start+1]
            op_start = gold_data["sentence"][:-1].find(".")
            op_end = gold_data["sentence"][:-1].rfind(".")

            if (
                not zero_shot
                and "contains " in gold
            ) or (zero_shot and "nothing" not in gold):
                new_pred = pred.replace("contains ", "")
                new_gold = gold.replace("contains ", "")
                if ("is empty" not in pred and "nothing" not in pred):
                    pred_items = set(
                        [
                            i.replace("the ", "").replace("the", "")
                            for i in re.split(r',? and |, ', new_pred)
                        ]
                    )
                else:
                    pred_items = set()

                gold_items = set(
                    [i.replace("the ", "") for i in new_gold.split(" and ")]
                )

                if pred != gold and pred_items == gold_items:
                    sorted_gold_list = sorted(list(gold_items))
                    if zero_shot:
                        pred = gold = "the " + \
                            " and the ".join(sorted_gold_list)
                    else:
                        pred = gold = "contains the " + " and the ".join(
                            sorted_gold_list
                        )

                if op_end != op_start:
                    op_string = gold_data["sentence"][op_start:op_end]
                    for gold_item in gold_items:
                        if gold_item.replace("_", " ") not in op_string:
                            initial_state_required = True
                        if "_" in gold_item:
                            gold_item_type = gold_item.split("_")[1]
                            if f"the {gold_item_type}" in op_string:
                                uses = re.findall(
                                    f"{_MODIFIERS_REGEX_STR} {gold_item_type}", gold_data["sentence"])
                                ambiguous_pragmatic_mention = ambiguous_pragmatic_mention or len(
                                    set(uses)) > 1
                            # else:
                            #    print(f"'{gold_item_type}' // {op_string}")

            move_ops1 = re.findall(
                f"Move the contents of Box [0-9] to Box {box_no}", gold_data["sentence"])
            # move_ops2 = re.findall(f"Move the contents of Box {box_no} to Box [0-9]", gold_data["sentence"])

            involves_move_content = len(move_ops1) > 0  # or len(move_ops2) > 0

            if pred.replace("the", "") == gold.replace(" the", ""):
                pred = pred.replace("the", "")
                gold = gold.replace(" the", "")
            initial_state_context = context.split(".")[0]
            initial_state = parse_intial_states(
                initial_state_context, zero_shot=zero_shot)
            prompt = context.split(".")[-2].strip()
            box_key = prompt.replace(" contains", "")
            total += 1

            # change format for zero-shot
            row["pred"] = pred
            row["gold"] = gold
            row["context"] = context

            row["initial_state_required"] = 1 if initial_state_required else 0
            row["ambiguous_pragmatic_mention"] = 1 if ambiguous_pragmatic_mention else 0
            row["involves_move_content"] = 1 if involves_move_content else 0

            row["hallucinated"] = 0
            if pred == gold:
                row["correct"] = 1
                row["precision"] = 1.0
                row["recall"] = 1.0
            else:
                row_state["correct"] = 0
                row["correct"] = 0

            if gold == initial_state[box_key]:
                row["eq_initial"] = 1
                row["initial_state_required"] = 1
            else:
                row["eq_initial"] = 0

            if gold == "is empty" or gold == "nothing":
                row["is_empty"] = 1
                row["num_obj"] = 0
                if pred != gold:
                    row["precision"] = 0.0
                    row["recall"] = 1.0
                    row["hallucinated"] = 1

            else:
                row["num_obj"] = gold.count(" and ") + 1
                row["is_empty"] = 0
                new_pred = pred.replace("contains ", "")
                new_gold = gold.replace("contains ", "")
                pred_items = set(
                    [
                        i.replace("the ", "").replace("the", "")
                        for i in new_pred.split(" and ")
                    ]
                )
                gold_items = set(
                    [i.replace("the ", "") for i in new_gold.split(" and ")]
                )
                tp = len(pred_items.intersection(gold_items))
                row["precision"] = float(tp) / len(pred_items)
                row["recall"] = float(tp) / len(gold_items)
                if row["precision"] < 1:
                    row["hallucinated"] = 1

            rows.append(row)
            if total % 7 == 0:
                rows_state.append(row_state)
                row_state = {"correct": 1}

    if gold_f is not None:
        gold_f.close()

    df = pd.DataFrame(rows)
    df_states = pd.DataFrame(rows_state)
    return df, df_states


def main():
    """
        Command line utility.
    """

    parser = argparse.ArgumentParser(
        "CLI utility for evaluating model outputs.")
    parser.add_argument("--model_output", type=str,
                        help="Path to model output in TSV format.", required=True)
    parser.add_argument("--gold_data", type=str,
                        help="Path to gold data in jsonl format.")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Set this when the data is in few-shot/zero-shot format.")

    args = parser.parse_args()

    res_ex, res_states = compute_metrics(
        args.model_output, gold_file_path=args.gold_data, zero_shot=args.zero_shot)

    acc = res_ex.agg(accuracy=pd.NamedAgg("correct", lambda x: x.mean()),
                     count=pd.NamedAgg("correct", lambda x: x.count()),
                     correct=pd.NamedAgg("correct", lambda x: x.sum())).transpose()
    acc_state = res_states.agg(accuracy=pd.NamedAgg("correct", lambda x: x.mean()),
                               count=pd.NamedAgg(
                                   "correct", lambda x: x.count()),
                               correct=pd.NamedAgg("correct", lambda x: x.sum())).transpose()

    print("#" * 80)
    print(f"{BOLD_START}Overall Accuracy:{BOLD_END}")
    print(
        f"Examples: {int(acc['correct'][0])}/{int(acc['count'][0])}={acc['accuracy'][0]}")
    print(
        f"States: {int(acc_state['correct'][0])}/{int(acc_state['count'][0])}={acc_state['accuracy'][0]}")

    acc_by_numops_local = res_ex.groupby("numops_local").agg(accuracy=pd.NamedAgg("correct", lambda x: x.mean()),
                                                         count=pd.NamedAgg("correct", lambda x: x.count()))    
    print("#" * 80)
    print(f"{BOLD_START}Example accuracy by number of operations affecting box state:{BOLD_END}")
    print(acc_by_numops_local)
    print("#" * 80)


if __name__ == "__main__":
    main()
