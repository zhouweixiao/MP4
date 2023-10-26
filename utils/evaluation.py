import os
import json
import rouge
import argparse


def prepare_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def evaluate(preds, tgts):
    for aggregator in ['Avg', 'Best']:
        print('Evaluation with {}'.format(aggregator))
        
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'
       
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2,
                                limit_length=True, length_limit=256,
                                length_limit_type='words',
                                apply_avg=apply_avg, apply_best=apply_best,
                                alpha=0.5, weight_factor=1.2, stemming=True)

        scores = evaluator.get_scores(preds, tgts)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            print(prepare_results(results['p'], results['r'], results['f'], metric))


def read_ground_truth(dataset_name):
    dataset_paths = {
        "SAMSum": "../datasets/Downstream_Datasets/SAMSum/test.json",
        "DIALOGSUM": "../datasets/Downstream_Datasets/DIALOGSUM/test.json",
        "TWEETSUMM": "../datasets/Downstream_Datasets/TWEETSUMM/test.json",
    }

    all_references = []
    dataset_path = dataset_paths.get(dataset_name)
    
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, "r") as rf:
            data = json.load(rf)
            if dataset_name == "SAMSum":
                all_references = [sample["reference_summ"] for sample in data]
            else:
                for sample in data:
                    summs = [value for key, value in sample.items() if "reference_summ" in key]
                    all_references.append(summs)
    else:
        print(f"Dataset {dataset_name} not found or path is incorrect.")

    return all_references


def read_txt_files(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    txt_files.sort(key=lambda x: int(x.split('.')[0]))

    content_list = []
    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content_list.append(content)
            
    return content_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_path", type=str, required=True, help="Path of the model prediction file")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["SAMSum", "DIALOGSUM", "TWEETSUMM"], help="Evaluation dataset name")

    args = parser.parse_args()

    preds = read_txt_files(args.pred_path)
    tgts = read_ground_truth(args.dataset_name)

    if preds and tgts:
        evaluate(preds, tgts)
    else:
        print("Either predictions or ground truth data is missing.")
