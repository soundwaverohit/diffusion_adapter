# evaluation/evaluate_results.py
"""
Evaluate and compare baseline vs. enhanced (diffusion-LM companion) outputs on GSM8K.

This script handles:
- Baseline JSONL where predictions are in the 'answer' field.
- Enhanced JSONL where predictions are in the 'prediction' field and references in 'reference'.

Usage:
    python evaluation/evaluate_results.py \
        --baseline baseline.jsonl \
        --enhanced results.jsonl \
        --top_n 5
"""
import argparse
import json
import re
from typing import List, Dict


def extract_first_number(s: str):
    """
    Extract the first integer or decimal from a string.
    Returns float if found, else None.
    """
    match = re.search(r"-?\d+\.?\d*", s)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def load_baseline(path: str) -> List[Dict]:
    """
    Load baseline JSONL which has 'question' and 'answer' fields.
    Returns list of dicts with keys: question, baseline_pred.
    """
    records = []
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            records.append({
                'question': rec['question'],
                'baseline_pred': rec.get('answer', '')
            })
    return records


def load_enhanced(path: str) -> Dict[str, Dict]:
    """
    Load enhanced JSONL which has 'question', 'prediction', 'reference'.
    Returns a dict mapping question -> record.
    """
    rec_map = {}
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            rec_map[rec['question']] = {
                'enh_pred': rec.get('prediction', ''),
                'reference': rec.get('reference', '')
            }
    return rec_map


def compute_accuracy(entries: List[Dict]) -> (int, int):
    """
    entries: list of dicts with keys: baseline_pred, enh_pred, reference.
    Returns: (baseline_correct, enhanced_correct, total)
    """
    b_corr = 0
    e_corr = 0
    total = len(entries)
    for e in entries:
        ref = e['reference']
        b_pred = e['baseline_pred']
        h_pred = e['enh_pred']

        ref_num = extract_first_number(ref)
        b_num = extract_first_number(b_pred)
        h_num = extract_first_number(h_pred)

        # compare baseline
        if ref_num is not None and b_num is not None:
            if abs(ref_num - b_num) < 1e-3:
                b_corr += 1
        else:
            if b_pred.strip() == ref.strip():
                b_corr += 1

        # compare enhanced
        if ref_num is not None and h_num is not None:
            if abs(ref_num - h_num) < 1e-3:
                e_corr += 1
        else:
            if h_pred.strip() == ref.strip():
                e_corr += 1

    return b_corr, e_corr, total


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs. enhanced results.")
    parser.add_argument('--baseline', required=True, help='Path to baseline JSONL')
    parser.add_argument('--enhanced', required=True, help='Path to enhanced JSONL')
    parser.add_argument('--top_n', type=int, default=5, help='Show N cases where enhanced correct but baseline wrong')
    args = parser.parse_args()

    # Load both
    baseline_list = load_baseline(args.baseline)
    enhanced_map = load_enhanced(args.enhanced)

    # Merge by question
    merged = []
    for b in baseline_list:
        q = b['question']
        if q not in enhanced_map:
            continue
        e = enhanced_map[q]
        merged.append({
            'question': q,
            'baseline_pred': b['baseline_pred'],
            'enh_pred': e['enh_pred'],
            'reference': e['reference']
        })

    # Compute accuracies
    b_corr, e_corr, total = compute_accuracy(merged)
    b_acc = b_corr / total * 100
    e_acc = e_corr / total * 100
    imp = e_acc - b_acc

    print(f"Baseline: {b_corr}/{total} = {b_acc:.2f}%")
    print(f"Enhanced: {e_corr}/{total} = {e_acc:.2f}%")
    print(f"Improvement: {imp:.2f} percentage points\n")

    # Display sample improvements
    print(f"Examples where baseline wrong but enhanced correct (up to {args.top_n}):\n")
    shown = 0
    for entry in merged:
        ref = entry['reference']
        ref_num = extract_first_number(ref)
        b_num = extract_first_number(entry['baseline_pred'])
        h_num = extract_first_number(entry['enh_pred'])

        base_ok = (ref_num is not None and b_num is not None and abs(ref_num - b_num) < 1e-3) or \
                  (ref_num is None and entry['baseline_pred'].strip()==ref.strip())
        enh_ok  = (ref_num is not None and h_num is not None and abs(ref_num - h_num) < 1e-3) or \
                  (ref_num is None and entry['enh_pred'].strip()==ref.strip())

        if not base_ok and enh_ok:
            print(f"Q: {entry['question']}")
            print(f"Ref: {ref}")
            print(f"Baseline prediction: {entry['baseline_pred']}")
            print(f"Enhanced prediction: {entry['enh_pred']}\n")
            shown += 1
            if shown >= args.top_n:
                break

if __name__ == '__main__':
    main()
