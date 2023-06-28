import argparse
import csv
import copy
import random
import json
import os
import re
import numpy as np

from numpy.random import poisson

from collections import Counter


# Possible operations
_OPERATIONS_DICT = {
    "move": "Move {content} {from_prep} Box {box1_num} to Box {box2_num}.",
    "remove": "Remove {content} from Box {box1_num}.",
    "put": "Put {content} into Box {box1_num}."
}

_OPERATIONS_DICT_ALT = {
    "move": "Pick up {content} {from_prep} Container {box1_num} and place {content_pronoun} into Container {box2_num}.",
    "remove": "Take {content} out of Container {box1_num}.",
    "put": "Place {content} inside Container {box1_num}.",
}

_ALT_BOX_NOUN = "Container"

_MODIFIERS = ["big", "small", "blue", "green", "red", "yellow"]

_SPLITS_PROP = {"train": 0.45, "dev": 0.1, "test": 0.45}


class WorldState:
    """
    A class representing a world state.
    """
    
    def __init__(
        self,
        all_objects,
        num_boxes,
        max_items_per_box,
        expected_num_items_per_box,
        contents=None,
        zero_shot=False,
    ):
        """Initialize WorldState.

        Args:
            all_objects (list): List of all possible objects.
            num_boxes (int): Number of boxes.
            max_items_per_box (int): Maximum number of objects per box.
            expected_num_items_per_box (int): Expected number of objects per box.
            contents (dict[list], optional): Initial contents of all boxes. Defaults to None.
            zero_shot (bool, optional): Whether to use zero-shot/in-context learning data format. Defaults to False.

        Raises:
            KeyError: Raised if invalid object is added to box.
            ValueError: Raised if too many objects are added to box.
        """
        self.boxes = [set([]) for _ in range(num_boxes)]
        self.all_objects = all_objects
        self.void = set(all_objects)
        self.num_boxes = num_boxes
        self.max_items_per_box = max_items_per_box
        self.expected_num_items_per_box = expected_num_items_per_box
        self.zero_shot = zero_shot

        if contents is not None:
            for i, b in enumerate(contents):
                for c in b:
                    if c not in self.all_objects:
                        raise KeyError(f"{c} is not a valid object!")
                if self.max_items_per_box > 0 and len(b) > self.max_items_per_box:
                    raise ValueError(
                        f"Attempted to add more than MAX_ITEMS_PER_BOX \
                         (={self.max_items_per_box}) items to box #{i}"
                    )
                self.void.difference_update(b)
                self.boxes[i].update(b)

    def __str__(self):
        ret = "Boxes:" + str(self.boxes) + "\n"
        ret += "Void:" + str(self.void)
        return ret

    def remove_from_box(self, box, content):
        """ Remove content from Box #box.

        Args:
            box (int): Box number.
            content (set): Set of objects to be removed. 

        Raises:
            KeyError: Raised if non-exstent object is removed.
        """
        if isinstance(content, (list, set)):
            for c in content:
                if c not in self.boxes[box]:
                    raise KeyError(f"{c} not in box #{box + 1}")
            self.boxes[box].difference_update(content)
            self.void.update(content)
        else:
            # throws KeyError if content is not in box
            self.boxes[box].remove(content)
            self.void.add(content)

    def add_to_box(self, box, content):
        if isinstance(content, list) or isinstance(content, set):
            for c in content:
                if c not in self.all_objects:
                    raise KeyError(f"{c} is not a valid object!")
                if c not in self.void:
                    raise KeyError(f"{c} is already in another box!")
            if (
                self.max_items_per_box > 0
                and (len(content) + len(self.boxes[box])) > self.max_items_per_box
            ):
                raise ValueError(
                    f"Attempted to add more than MAX_ITEMS_PER_BOX \
                    (={self.max_items_per_box}) items to box #{box}"
                )
            self.void.difference_update(content)
            self.boxes[box].update(content)
        else:
            if (
                self.max_items_per_box > 0
                and (1 + len(self.boxes[box])) > self.max_items_per_box
            ):
                raise ValueError(
                    f"Attempted to add more than MAX_ITEMS_PER_BOX \
                    (={self.max_items_per_box}) items to box #{box}"
                )
            self.void.remove(content)
            self.boxes[box].add(content)

    def move_to_box(self, from_box, to_box, content):
        if isinstance(content, list) or isinstance(content, set):
            for c in content:
                if c not in self.boxes[from_box]:
                    raise KeyError(f"{c} not in box #{from_box}")
            if (
                self.max_items_per_box > 0
                and (len(content) + len(self.boxes[to_box])) > self.max_items_per_box
            ):
                raise ValueError(
                    f"Attempted to add more than MAX_ITEMS_PER_BOX \
                    (={self.max_items_per_box}) items to box #{to_box}"
                )
            self.boxes[from_box].difference_update(content)
            self.boxes[to_box].update(content)
        else:
            if (
                self.max_items_per_box > 0
                and (1 + len(self.boxes[to_box])) > self.max_items_per_box
            ):
                raise ValueError(
                    f"Attempted to add more than MAX_ITEMS_PER_BOX \
                    (={self.max_items_per_box}) items to box #{to_box}"
                )
            # throws KeyError if content is not in from_box
            self.boxes[from_box].remove(content)
            self.boxes[to_box].add(content)

    def empty_box(self, box):
        raise NotImplementedError
        # if len(self.boxes[box]) < 1:
        #    raise Exception(f"Box {box} is already empty!")
        # self.void.update(self.boxes[box])
        # self.boxes[box].clear()

    @staticmethod
    def sample_initial_world_state(
        all_objects,
        num_boxes,
        max_items_per_box,
        expected_num_items_per_box,
        zero_shot=False,
    ):
        s = WorldState(
            all_objects,
            num_boxes,
            max_items_per_box,
            expected_num_items_per_box,
            zero_shot=zero_shot,
        )

        num_items = poisson(expected_num_items_per_box, num_boxes)

        while sum(num_items) > len(all_objects):
            num_items = poisson(expected_num_items_per_box, num_boxes)

        if max_items_per_box > 0:
            num_items = np.minimum(num_items, [max_items_per_box])

        for i, n in enumerate(num_items):
            items = np.random.choice(list(s.void), n, replace=False)
            s.add_to_box(i, list(items))

        return s

    def state_description(self, box=None, alt_description=False, box_noun="Box"):
        if box is not None:
            s = self._describe_box(
                box, individual=True, alt_description=alt_description, box_noun=box_noun
            )
            return s[0].upper() + s[1:]
        else:
            s = ", ".join(
                [
                    self._describe_box(
                        b,
                        individual=False,
                        alt_description=alt_description,
                        box_noun=box_noun,
                    )
                    for b in range(self.num_boxes)
                ]
            )
            return s[0].upper() + s[1:] + "."

    def _describe_box(
        self, box, individual=True, alt_description=False, box_noun="Box"
    ):
        box_name = str(box)
        if box_noun == "Container":
            box_name = chr(box + 65)
        # first_char = "T" if individual else "t" # capitalize t if not part of an enumeration
        final_char = (
            "." if individual else ""
        )  # add period if not part of an enumeration
        if len(self.boxes[box]) == 0:
            if alt_description:
                return f"there is nothing in {box_noun} {box_name}{final_char}"
            else:
                if self.zero_shot:
                    return f"{box_noun} {box_name} contains nothing{final_char}"
                else:
                    return f"{box_noun} {box_name} is empty{final_char}"
                # return f"{first_char}he {box_name} box is empty{final_char}"
        elif len(self.boxes[box]) == 1:
            if alt_description:
                return f"the {list(self.boxes[box])[0]} is in {box_noun} {box_name}{final_char}"
            else:
                return f"{box_noun} {box_name} contains the {list(self.boxes[box])[0]}{final_char}"

            # return f"{first_char}he {box_name} box contains the {list(self.boxes[box])[0]}{final_char}"
        else:
            box_contents = " and ".join([f"the {c}" for c in sorted(self.boxes[box])])
            if alt_description:
                return f"{box_contents} are in {box_noun} {box_name}{final_char}"
            else:
                return f"{box_noun} {box_name} contains {box_contents}{final_char}"
            # return f"{first_char}he {box_name} box contains {box_contents}{final_char}"

    def __eq__(self, o):
        for box1, box2 in zip(self.boxes, o.boxes):
            if box1 != box2:
                return False

        return True

    def __hash__(self):
        sub_hashes = []
        for box in self.boxes:
            sub_hashes.append(hash(tuple(box)))

        return hash(tuple(sub_hashes))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_vocabulary_file",
        type=str,
        default="data/objects_with_bnc_frequency.csv",
        help='Path to a .csv file with a string field "object_names".',
    )
    parser.add_argument(
        "--disjoint_object_vocabulary_file",
        type=str,
        default=None,
        help='Path to a .csv file with a string field "object_names" \
              that will be used to create a test set with disjoint item names. (Splits can be specified with --disjoint_object_splits, default: test) ',
    )
    parser.add_argument(
        "--disjoint_object_splits",
        type=str,
        default="test",
        help="Splits for which data sets with disjoint item names should be created.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/boxes",
        help="Path to a directory where the sampled dataset will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default="2255",
    )
    parser.add_argument(
        "--num_boxes", type=int, default=7, help="Number of boxes in the world."
    )
    parser.add_argument(
        "--expected_num_items_per_box",
        type=int,
        default=1,
        help="Expected number of items per box.",
    )
    parser.add_argument(
        "--max_items_per_box",
        type=int,
        default=1,
        help="Maximum number of items per box.",
    )
    parser.add_argument("--num_samples", type=int, default=2200)
    parser.add_argument(
        "--num_operations",
        type=int,
        default=10,
        help="Total number of operations in a single sample.",
    )
    parser.add_argument(
        "--disjoint_numops",
        action="store_true",
        help="If set, the test set will contain operation sequences sampled with num_operations+10.",
    )

    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="If set, the context includes 'contains' and 'is empty' is represented as 'contains nothing'.",
    )

    parser.add_argument(
        "--include_modifiers",
        type=str,
        default="never",
        choices=["never", "test", "always"],
        help="If set, object descriptions include adjectival modifiers for the objects.",
    )

    parser.add_argument(
        "--omit_modifiers_in_ops",
        type=str,
        default="never",
        choices=["never", "test", "always"],
        help="If set, object descriptions include adjectival modifiers only in description of operations when necessary for disambiguation.",
    )

    parser.add_argument(
        "--alternative_forms",
        type=str,
        default="never",
        choices=["never", "train", "test", "always"],
        help="If not set to never, initial description and operations use alternative formulations (either in all splits or just the test split).",
    )

    parser.add_argument("--rarify", action="store_true")

    parser.add_argument(
        "--all_contents_operation",
        action="store_true",
        help="If set, use the operation 'Move the contents' instead of enumerating the contents when all the contents are being moved.",
    )

    return parser.parse_args()


def load_objects_from_csv(csv_path):
    object_list = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_list.append(row["object_name"])
    return frozenset(object_list)


def disjoint_object_map(object_set, disjoint_name_csv_path):
    disjoint_obj_list = []
    with open(disjoint_name_csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disjoint_obj_list.append(row["object_name"])
    assert len(object_set) == len(disjoint_obj_list)

    obj_map = {obj: dobj for obj, dobj in zip(object_set, disjoint_obj_list)}
    return obj_map


def random_nonempty_subset(s):
    """
        Returns a non-empty subset of set s.
    """
    if len(s) < 1:
        raise ValueError("Input set cannot be empty!")
    out = set()
    for el in s:
        # random coin flip
        if random.randint(0, 1) == 0:
            out.add(el)
    if len(out) < 1:
        out = random_nonempty_subset(s)
    return out



def describe_operation(
    op, box1, box2, contents, alt_description=False, all_contents=False
):
    """
    Describe operation based on operation type, box 1, box 2 (optional)
    and the contents that should be added/removed/moved (optional).

    Args:
        op (str): identifer of operation
        box1 (int): first box the operation is acting upon
        box2 (int): second box the operation is acting upon (set to None for unary operations)
        contents (iterable): contents that are added/moved/removed from boxes
        alt_description (bool, optional): Use alternative descriptions for operations. Defaults to False.
        all_contents (bool, optional): Use special "Move all contents" operation when all contents are moved. Defaults to False.

    Returns:
        str: description of the operation
    """
    s = _OPERATIONS_DICT[op] if not alt_description else _OPERATIONS_DICT_ALT[op]
    content_str, box1_num, box2_num, content_pronoun, content_verb, from_prep = (
        None,
        None,
        None,
        None,
        None,
        None
    )
    if all_contents and op == "move":
        content_str = "the contents"
        content_pronoun = "them"
        from_prep = "of"
    elif contents is not None and len(contents) > 0:
        content_str = " and ".join([f"the {c}" for c in sorted(contents)])
        content_pronoun = "it" if len(contents) == 1 else "them"
        content_verb = "is" if len(contents) == 1 else "are"
        from_prep = "from" if not alt_description else "in"
    if box1 is not None:
        box1_num = str(box1) if not alt_description else chr(box1 + 65)
    if box2 is not None:
        box2_num = str(box2) if not alt_description else chr(box2 + 65)
    op_str = s.format(
        box1_num=box1_num,
        box2_num=box2_num,
        content=content_str,
        content_pronoun=content_pronoun,
        content_verb=content_verb,
        from_prep=from_prep
    )
    return op_str[0].upper() + op_str[1:]


def example_to_t5(ex, zero_shot=False, modifier_map=None, pragmatic=False):
    """ 
    Turns example into format used to train/test T5 models.

    Args:
        ex (str): Description of initial state and operations.
        zero_shot (bool, optional): Output zero-shot/in-context 
            learning format. Defaults to False.
        modifier_map (dict, optional): Map to replace object names with modified 
            object names. Defaults to None.
        pragmatic (bool, optional): Whether modifiers should be omitted. Defaults to False.

    Returns:
        dict: Object with original sentence, masked sentence, and masked content.
    """
    sentences = ex.split(".")

    # replace non-modified object descriptions
    # with modified object descriptions
    if modifier_map is not None:
        for key, val in modifier_map.items():
            pattern = r"(\sthe\s)" + key + r"([,.\s]|$)"
            repl = r"\1" + val + r"\2"
            if pragmatic:
                sentences[0] = re.sub(pattern, repl, sentences[0])
                sentences[-2] = re.sub(pattern, repl, sentences[-2])
            else:
                ex = re.sub(pattern, repl, ex)

    if not pragmatic:
        sentences = ex.split(".")

    last_sent = sentences[-2]

    if "is empty" in last_sent:
        masked_sentence = (
            ".".join(sentences[0:-2])
            + "."
            + last_sent.replace("is empty", "<extra_id_0> .")
        )
        masked_content = "<extra_id_0> is empty"
    else:
        start = last_sent.index("contains")
        if zero_shot:
            start += len("contains ")
        masked_sentence = (
            ".".join(sentences[0:-2]) + "." + last_sent[0:start] + "<extra_id_0> ."
        )
        masked_content = "<extra_id_0> " + last_sent[start:]

    return {
        "sentence": ".".join(sentences).lstrip(),
        "sentence_masked": masked_sentence.lstrip(),
        "masked_content": masked_content.lstrip(),
    }


def sample_operation_sequences(
    world_state,
    operations,
    box_names,
    num_operations,
    generate_alternative_forms=False,
    all_contents_operation=False,
):
    """Performs operation sequence sampling.

    Args:
        world_state: Initial WorldState.
        operations: A list of strings describing possible operations.
        box_names: A list of box names in the world.
        num_operations: Total number of operations in a single sequence.
        generate_alternative_forms: Whether to generate altenatively phrased descriptions.
        all_contents_operation: Whether to generate operations of the form "Move the contents from Box X to Box Y."

    Returns:
        operation_sequence: A list of sampled operations.
        world_states: A list of WorldStates after each sampled operation.
        alt_operation_sequence: A list of alternatively phrased operations.
    """
    operation_sequence = []
    world_states = []

    if generate_alternative_forms:
        alt_operation_sequences = []

    # Convert initial world state into naturalistic language
    # and append to operation_sequence
    world_states.append(copy.deepcopy(world_state))
    operation_sequence.append(world_state.state_description())
    if generate_alternative_forms:
        alt_operation_sequences.append(
            world_state.state_description(
                box=None, alt_description=True, box_noun=_ALT_BOX_NOUN
            )
        )
    for _ in range(num_operations):
        op = None
        box1 = None
        box2 = None
        contents = []
        all_contents = False
        while True:
            # Sample an operation (each operation has different arity)
            op = random.choice(operations)
            box1 = random.choice(box_names)
            all_contents = False
            if op == "empty":
                try:
                    world_state.empty_box(box1)
                    break
                except (ValueError, KeyError):
                    continue
            elif op == "move":
                if len(world_state.boxes[box1]) < 1:
                    continue
                box2 = random.choice(box_names[0:box1] + box_names[box1 + 1 :])
                contents = random_nonempty_subset(world_state.boxes[box1])
                if all_contents_operation and len(contents) == len(
                    world_state.boxes[box1]
                ):
                    all_contents = True

                try:
                    world_state.move_to_box(box1, box2, contents)
                    break
                except (ValueError, KeyError):
                    continue
            elif op == "put":
                exp_value = max(1, int(world_state.expected_num_items_per_box / 2))
                no_items = poisson(exp_value)
                if (
                    world_state.max_items_per_box > 0
                    and (no_items + len(world_state.boxes[box1]))
                    > world_state.max_items_per_box
                ) or no_items < 1:
                    continue

                contents = random.sample(list(world_state.void), no_items)

                try:
                    world_state.add_to_box(box1, contents)
                    break
                except (ValueError, KeyError):
                    continue
            elif op == "remove":
                if len(world_state.boxes[box1]) < 1:
                    continue

                contents = random_nonempty_subset(world_state.boxes[box1])
                try:
                    world_state.remove_from_box(box1, contents)
                    break
                except (ValueError, KeyError):
                    continue
            else:
                continue

        world_states.append(copy.deepcopy(world_state))
        operation_sequence.append(
            describe_operation(op, box1, box2, contents, all_contents=all_contents)
        )
        if generate_alternative_forms:
            alt_operation_sequences.append(
                describe_operation(
                    op,
                    box1,
                    box2,
                    contents,
                    alt_description=True,
                    all_contents=all_contents,
                )
            )

    if generate_alternative_forms:
        return operation_sequence, world_states, alt_operation_sequences
    return operation_sequence, world_states, None


def check_state_signature(state, state_signature_set, max_items_per_box=9):
    """Checks whether a state's signature is part of state_signature_set.

    Args:
        state (WorldState): A WorldState.
        state_signature_set (set): A set of state signatures.
        max_items_per_box (int, optional): Maximum number of items per box, 
            used to compute signatures. Defaults to 9.

    Returns:
        bool: True if state's signature is in state_signature_set.
    """

    base = max_items_per_box + 1
    state_signature = 0
    for box in state.boxes:
        state_signature *= base
        state_signature += len(box)
    return state_signature in state_signature_set


def make_modifier_map(object_set, pragmatic=False):
    """Constructs modifier map for pragmatic datasets.

    Args:
        object_set (set): A set of object names.
        pragmatic (bool, optional): If set to true. Defaults to False.

    Returns:
        dict: A map from original object names to modified object names.
    """
    modifier_map = {}
    if not pragmatic:
        for obj in object_set:
            mod = random.choice(_MODIFIERS)
            modifier_map[obj] = f"{mod} {obj}"
    else:
        sampled_objects = random.sample(
            object_set, len(object_set) // len(_MODIFIERS) + 1
        )
        new_objects = []
        for obj in sampled_objects:
            for mod in _MODIFIERS:
                new_objects.append(f"{mod} {obj}")
        for obj in object_set:
            modifier_map[obj] = new_objects.pop()
    return modifier_map


def non_unique_object_types(modified_objects):
    """Returns a set of object types that are not
       unique in modified_objects.

    Args:
        modified_objects (set): A set of strings with modified object names.

    Returns:
        set: A set of object types that appear more than once.
    """
    types = [o.split(" ")[-1] for o in modified_objects]
    counts = Counter(types)
    non_uniques = []
    for k, c in counts.items():
        if c > 1:
            non_uniques.append(k)
    return set(non_uniques)


def pragmatify(state, operation, modifier_map):
    """Remove modifies from operation if pragmatically unncessary.

    Args:
        state (WorldState): A WorldState
        operation (str): A description of the operation.
        modifier_map (dict): A map from unmodified to modified object names.

    Raises:
        ValueError: Raised if operation is unknown / described in unknown format.

    Returns:
        str: "Pragmatified" operation description.
    """
    toks = operation[:-1].split(" ")

    if modifier_map is not None and toks[0] == "Put":
        contents_descr = " ".join(toks[1:-3])
        contents = [
            "the " + modifier_map[s.replace("the ", "")]
            for s in contents_descr.split(" and ")
        ]
        return toks[0] + " " + " and ".join(contents) + " " + " ".join(toks[-3:]) + "."
    elif toks[0] == "Put":
        return operation

    if modifier_map is not None:
        state = copy.deepcopy(state)
        state.void = set([modifier_map[obj] for obj in state.void])
        for i, box in enumerate(state.boxes):
            state.boxes[i] = set([modifier_map[obj] for obj in box])

    if toks[0] == "Move":
        contents_descr = " ".join(toks[1:-6])
        if contents_descr == "the contents":
            return operation
        src_box = int(toks[-4])
    elif toks[0] == "Remove":
        contents_descr = " ".join(toks[1:-3])
        src_box = int(toks[-1])
    else:
        raise ValueError(f"Unknown operation: {toks[0]}; Operation {operation}")

    non_unique_types = non_unique_object_types(state.boxes[src_box])

    contents = [s.replace("the ", "") for s in contents_descr.split(" and ")]
    if modifier_map is not None:
        contents = [modifier_map[c] for c in contents]
    new_contents = []
    for obj_descr in contents:
        obj_type = obj_descr.split(" ")[-1]
        if obj_type in non_unique_types:
            new_contents.append(f"the {obj_descr}")
        else:
            new_contents.append(f"the {obj_type}")

    if toks[0] == "Move":
        return (
            toks[0] + " " + " and ".join(new_contents) + " " + " ".join(toks[-6:]) + "."
        )
    elif toks[0] == "Remove":
        return (
            toks[0] + " " + " and ".join(new_contents) + " " + " ".join(toks[-3:]) + "."
        )
    else:
        return operation


def main(args):
    """
        Main function.
    """
    print(args)

    assert args.max_items_per_box < 10, "max_items_per_box cannot be greater than 9."

    random.seed(args.seed)
    np.random.seed(args.seed)

    objects_set = load_objects_from_csv(args.object_vocabulary_file)

    if args.disjoint_object_vocabulary_file is not None:
        obj_map = disjoint_object_map(objects_set, args.disjoint_object_vocabulary_file)

    if not args.rarify:
        operations = list(_OPERATIONS_DICT.keys())
        box_names = list(range(0, args.num_boxes))

        sampled_sequences = []

        # sampling twice as many sequences as requested since there may not be
        # uniform distribution across all binary state types or
        # duplicate initial states
        num_samples = args.num_samples * 2
        max_num_operations = (
            args.num_operations + 10 if args.disjoint_numops else args.num_operations
        )

        generate_alt_descriptions = args.alternative_forms != "never"

        for _ in range(num_samples):
            initial_world_state = WorldState.sample_initial_world_state(
                all_objects=objects_set,
                num_boxes=args.num_boxes,
                max_items_per_box=args.max_items_per_box,
                expected_num_items_per_box=args.expected_num_items_per_box,
                zero_shot=args.zero_shot,
            )
            (
                operation_sequence,
                world_states,
                operation_sequence_alt,
            ) = sample_operation_sequences(
                initial_world_state,
                operations,
                box_names,
                max_num_operations,
                generate_alternative_forms=generate_alt_descriptions,
                all_contents_operation=args.all_contents_operation,
            )
            sampled_sequences.append(
                (world_states, operation_sequence, operation_sequence_alt)
            )

        print(
            f"finished sampling {num_samples}",
            f"sequences of length {max_num_operations}.",
        )

        state_signatures_count = (args.max_items_per_box + 1) ** args.num_boxes
        state_signatures_all = list(range(state_signatures_count))

        random.shuffle(state_signatures_all)

        splits_size = {
            split: int(args.num_samples * prop) for split, prop in _SPLITS_PROP.items()
        }

        state_signature_sets = {
            "train": set(
                state_signatures_all[
                    0 : int(state_signatures_count * _SPLITS_PROP["train"])
                ]
            ),
            "dev": set(
                state_signatures_all[
                    int(state_signatures_count * _SPLITS_PROP["train"]) : int(
                        state_signatures_count
                        * (_SPLITS_PROP["train"] + _SPLITS_PROP["dev"])
                    )
                ]
            ),
            "test": set(
                state_signatures_all[
                    -int(state_signatures_count * _SPLITS_PROP["test"]) :
                ]
            ),
        }
        # print(binary_states)
        # store all initial states to make sure we don't have exact duplicates
        existing_states = set()

        count_num = {split: 0 for split in splits_size}
        os.makedirs(args.output_dir, exist_ok=True)
        out_files = {
            split: open(os.path.join(args.output_dir, f"{split}-t5.jsonl"), "w")
            for split in splits_size
        }
        num_operations = {
            split: args.num_operations
            if split == "train" or not args.disjoint_numops
            else args.num_operations + 10
            for split in splits_size
        }

        # Save to file
        num_obj_per_box_counts = Counter()

        for i, (states, ops, ops_alt) in enumerate(sampled_sequences):
            # check for exact duplicate
            if states[0] in existing_states:
                print("Skipped exact duplicate of initial state!")
                continue

            write_example = False

            # find correct bucket
            for split, size in splits_size.items():
                if count_num[split] < size and check_state_signature(
                    states[0],
                    state_signature_sets[split],
                    max_items_per_box=args.max_items_per_box,
                ):
                    count_num[split] = count_num[split] + 1
                    existing_states.add(states[0])
                    out_f = out_files[split]
                    write_example = True
                    break

            if not write_example:
                # no bucket found
                continue

            modifier_map = None
            pragmatic = args.omit_modifiers_in_ops in ["always", split]
            if args.include_modifiers in ["always", split]:
                modifier_map = make_modifier_map(objects_set, pragmatic=pragmatic)

            prefix = ""
            prev_state = None
            box_noun = "Box"
            alt_descriptions = args.alternative_forms in ["always", split]
            if alt_descriptions:
                ops = ops_alt
                box_noun = _ALT_BOX_NOUN
            numops = [0 for _ in range(args.num_boxes)]
            numops_by_type = [
                {t: 0 for t in _OPERATIONS_DICT.keys()} for _ in range(args.num_boxes)
            ]
            for j, (state, op) in enumerate(zip(states, ops)):
                if j > num_operations[split]:
                    break

                if j > 0 and pragmatic:
                    op = pragmatify(prev_state, op, modifier_map)

                prefix += " " + op
                op_type = op.split()[0].lower()
                if prev_state is not None and op_type not in numops_by_type[0]:
                    if op_type in ["pick", "take", "place"]:  # alternative forms
                        if op_type == "pick":
                            op_type = "move"
                        elif op_type == "take":
                            op_type = "remove"
                        elif op_type == "place":
                            op_type = "put"
                    else:
                        raise ValueError(f"Invalid operation type: {op}")
                for box in range(args.num_boxes):
                    # increase numops if contents of boxes changed compared to previous state
                    if (
                        prev_state is not None
                        and prev_state.boxes[box] != state.boxes[box]
                    ):
                        numops[box] += 1
                        numops_by_type[box][op_type] += 1

                    ex = prefix + " " + state.state_description(box, box_noun=box_noun)
                    out_d = example_to_t5(
                        ex,
                        zero_shot=state.zero_shot,
                        modifier_map=modifier_map,
                        pragmatic=pragmatic,
                    )
                    if "contains" in out_d["masked_content"] or (
                        state.zero_shot and "nothing" not in out_d["masked_content"]
                    ):
                        num_objs = out_d["masked_content"].count(" and ") + 1
                    else:
                        num_objs = 0

                    num_obj_per_box_counts[num_objs] += 1

                    out_d["sample_id"] = i
                    out_d["numops"] = numops[box]
                    out_d["numops_by_op"] = numops_by_type[box]
                    out_f.write(json.dumps(out_d) + "\n")
                prev_state = state

        for split, size in splits_size.items():
            out_files[split].close()
            assert count_num[split] == size, (split, count_num[split], size)

    if args.disjoint_object_vocabulary_file is not None:
        splits = args.disjoint_object_splits.split(",")
        for split in splits:
            with open(
                os.path.join(args.output_dir, f"{split}-disjoint-vocab-t5.jsonl"), "w"
            ) as disjoint_f, open(
                os.path.join(args.output_dir, f"{split}-t5.jsonl")
            ) as test_f:
                for line in test_f:
                    for key, val in obj_map.items():
                        pattern = r"(\s)" + key + r'([,\.\s"])'
                        repl = r"\1" + val + r"\2"
                        line = re.sub(pattern, repl, line)
                    disjoint_f.write(line)

    print(f"Saved outputs to {args.output_dir}.")

    if not args.rarify:
        print("Distribution of num objects per box:")
        for k in sorted(num_obj_per_box_counts.keys()):
            print(k, num_obj_per_box_counts[k])


if __name__ == "__main__":
    args = parse_args()

    # pragmatify currently not compatible with alternative formulations
    if args.omit_modifiers_in_ops != "never" and args.alternative_forms != "never":
        raise argparse.ArgumentError(
            None,
            message="--omit_modifiers_in_ops cannot be used together with --alternative_forms",
        )

    #    test(args)
    main(args)
