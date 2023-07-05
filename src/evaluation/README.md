### Evaluation code

`compute_metrics.py` computes the accuracy for each example. 

The model output to be evaluated should be in the same TSV format as the example model outputs:

```
	target	prediction	input
0	contains the machine	contains the ticket	Box 0 contains the ticket, Box 1 contains the leaf and the stone and the wire, Box 2 contains the coat, Box 3 contains the brain, Box 4 contains the brick and the cake and the glass, Box 5 contains the chemical, Box 6 is empty. Box 0
````

Basic statistics such as the overall accuracy and the accuracy as a function of the number of operations acting upon a box can be computed with the command line tool:

```
python compute_metrics.py --model_output <PATH-TO-MODEL-OUTPUT.tsv> --gold_data <PATH-TO-GOLD-DATA.jsonl>
```

To compute metrics on specific subsets (e.g., all empty boxes), modify the main function in `evaluation_metrics.py` or call `compute_metrics()` from another script.

