import argparse
import json
from nltk.translate.bleu_score import sentence_bleu


def compute_bleu_with_ref_map(result_path: str, ref_map: dict, id_key: str, hyp_key: str, weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    correct = 0.0
    cnt = 0
    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        for item in results:
            image_id = str(item[id_key])
            if image_id not in ref_map:
                continue
            ref_text = ref_map[image_id]
            score = sentence_bleu([ref_text], item[hyp_key], weights=weights)
            correct += score
            cnt += 1
    return correct / max(1, cnt)


def compute_bleu_uniform(result_path: str, reference_text: str, hyp_key: str, weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    correct = 0.0
    cnt = 0
    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        for item in results:
            score = sentence_bleu([reference_text], item[hyp_key], weights=weights)
            correct += score
            cnt += 1
    return correct / max(1, cnt)


def parse_weights(arg: str) -> tuple:
    parts = [p.strip() for p in arg.split(",")]
    nums = tuple(float(p) for p in parts)
    return nums


def main():
    parser = argparse.ArgumentParser(description="Evaluate caption results with BLEU using captions_clean.json or a universal reference.")
    parser.add_argument("--result_dir", required=True, type=str, help="Directory with captions_clean.json, captions_adv.json, captions_purify.json")
    parser.add_argument("--id_key", type=str, default="image_id", help="Key for image id in results")
    parser.add_argument("--hyp_key", type=str, default="caption", help="Key for hypothesis caption in results")
    parser.add_argument("--bleu4_weights", type=str, default="0.25,0.25,0.25,0.25", help="BLEU-4 n-gram weights, comma-separated")
    parser.add_argument("--universal_reference", type=str, default=None, help="If provided, use this single reference text for all samples (UAP setting)")
    args = parser.parse_args()

    bleu4_weights = parse_weights(args.bleu4_weights)
    paths = {
        "clean": f"{args.result_dir}/captions_clean.json",
        "adv": f"{args.result_dir}/captions_adv.json",
        "purify": f"{args.result_dir}/captions_purify_mask.json",
    }

    use_universal = args.universal_reference is not None
    if not use_universal:
        # Build reference map from clean results
        with open(paths["clean"], 'r', encoding='utf-8') as f:
            clean_items = json.load(f)
        ref_map = {str(it[args.id_key]): it[args.hyp_key] for it in clean_items}

    for name in ["adv", "purify"]:
        path = paths[name]
        if use_universal:
            bleu4 = compute_bleu_uniform(path, args.universal_reference, hyp_key=args.hyp_key, weights=bleu4_weights)
            bleu1 = compute_bleu_uniform(path, args.universal_reference, hyp_key=args.hyp_key, weights=(1.0, 0.0, 0.0, 0.0))
        else:
            bleu4 = compute_bleu_with_ref_map(path, ref_map, id_key=args.id_key, hyp_key=args.hyp_key, weights=bleu4_weights)
            bleu1 = compute_bleu_with_ref_map(path, ref_map, id_key=args.id_key, hyp_key=args.hyp_key, weights=(1.0, 0.0, 0.0, 0.0))
        print(f"[{name}] BLEU-4 = {bleu4:.4f} | BLEU-1 = {bleu1:.4f}")


if __name__ == "__main__":
    main()


