"""Reproduce the paper's Tables 6-9 (secondary selection, selector and classifier comparisons).

For every subject (and Dataset 2 class pair) and each proposed feature extractor
(CSP-Wavelet, CSP-WPD, CSP-FB) this fits LASSO and LOG once and evaluates:
  * Table 6/7: FLDA on all LASSO/LOG-selected features (|w| > 0, "without secondary
    feature selection") vs. the threshold ensemble's best model ("with").
  * Table 8:   F-score, GA and BPSO selection + FLDA, next to LASSO and LOG.
  * Table 9:   the LOG threshold ensemble with linear SVM, BLDA or FLDA.
Per-subject results are cached in ``results/ablation/cache`` so an interrupted run
resumes. Tables are laid out as in the paper (grouped headers) and written to
``results/ablation/`` as ablation_tables.html, .tex (booktabs), .md and one .csv
per table. "All data" pools every subject.

Run from the project root:
    python src/results_generation/ablation_tables.py            # all datasets
    python src/results_generation/ablation_tables.py --jobs 4 --dataset 4
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers.blda import BLDA
from src.classifiers.ensemble_flda import ThresholdEnsembleFLDA
from src.data.data_loader import iter_dataset_trials
from src.features.csp_fb import extract_train_test as csp_fb
from src.features.csp_wavelet import extract_train_test as csp_wavelet
from src.features.csp_wpd import extract_train_test as csp_wpd
from src.feature_selection.lasso_selection import LassoFeatureSelector
from src.feature_selection.log_selection import LogFeatureSelector
from src.feature_selection.wrapper_selection import BPSOSelector, FScoreSelector, GASelector

EXTRACTORS = {"CSP-Wavelet": csp_wavelet, "CSP-WPD": csp_wpd, "CSP-FB": csp_fb}
DATASETS = {"1": "Dataset 1", "2a": "Dataset 2", "2b": "Dataset 3", "4": "Dataset 4"}
OUT = PROJECT_ROOT / "results" / "ablation"


def _flda_on(support, x_train, y_train, x_test, y_test):
    if not np.any(support):
        return 0.0
    model = LinearDiscriminantAnalysis(solver="svd").fit(x_train[:, support], y_train)
    return 100.0 * accuracy_score(y_test, model.predict(x_test[:, support]))


def _ensemble(selector, x_train, y_train, x_test, y_test, make_model=None):
    ensemble = ThresholdEnsembleFLDA(make_model=make_model).fit(
        selector.transform(x_train), y_train, selector.coef_)
    return ensemble.evaluate(selector.transform(x_test), y_test)


def evaluate_unit(trial, extractor_name):
    """All Table 6-9 accuracies (%) for one subject/pair and one feature extractor."""
    y_tr, y_te = trial.y_train, trial.y_test
    x_tr, x_te, _ = EXTRACTORS[extractor_name](trial.X_train, y_tr, trial.X_test, trial.fs)
    out = {}
    for name, selector in (("LASSO", LassoFeatureSelector()), ("LOG", LogFeatureSelector())):
        selector.fit(x_tr, y_tr)
        result = _ensemble(selector, x_tr, y_tr, x_te, y_te)
        out[f"{name}_without"] = 100.0 * result["threshold_results"][0]["accuracy"]  # |w| > 0 only
        out[f"{name}_with"] = 100.0 * result["paper_score"]
        if name == "LOG":
            out["LOG_SVM"] = 100.0 * _ensemble(selector, x_tr, y_tr, x_te, y_te,
                                               lambda: SVC(kernel="linear", C=1.0))["paper_score"]
            out["LOG_BLDA"] = 100.0 * _ensemble(selector, x_tr, y_tr, x_te, y_te, BLDA)["paper_score"]
    for name, selector in (("F-score", FScoreSelector()), ("GA", GASelector()), ("BPSO", BPSOSelector())):
        out[name] = _flda_on(selector.fit(x_tr, y_tr).get_support(), x_tr, y_tr, x_te, y_te)
    return out


def _cache_path(dataset, trial, extractor_name):
    unit = trial.subject + (f"_{trial.pair}" if trial.pair else "")
    return OUT / "cache" / f"dataset_{dataset}" / extractor_name / f"{unit}.json"


def _run_task(dataset, trial, extractor_name):
    path = _cache_path(dataset, trial, extractor_name)
    if path.exists():
        return
    result = evaluate_unit(trial, extractor_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    print(f"done {path.relative_to(OUT / 'cache')}", flush=True)


def load_results(datasets):
    """{(dataset, extractor): {metric: [per-unit accuracies]}} from the cache."""
    collected = {}
    for dataset in datasets:
        for extractor_name in EXTRACTORS:
            folder = OUT / "cache" / f"dataset_{dataset}" / extractor_name
            per_metric = {}
            for path in sorted(folder.glob("*.json")):
                for metric, value in json.loads(path.read_text(encoding="utf-8")).items():
                    per_metric.setdefault(metric, []).append(value)
            collected[(dataset, extractor_name)] = per_metric
    return collected


def _mean(collected, datasets, extractor_name, metric, dataset=None):
    keys = [dataset] if dataset else datasets  # None -> pool all datasets ("All data")
    values = sum((collected[(d, extractor_name)].get(metric, []) for d in keys), [])
    return f"{np.mean(values):.2f}" if values else "n/a"


# ----------------------------------------------------------------------------------
# Table layout. Each table is a dict:
#   number, name, caption
#   stub:   header(s) of the leading label column(s), e.g. ["Datasets"]
#   groups: [(group heading or None, [column headings])]  -> merged header row
#   rows:   [[stub cell(s)..., value cells...]]
# A repeated first stub cell (Table 8's feature extraction) is written once per block.
# ----------------------------------------------------------------------------------
def _columns(table):
    return [col for _, cols in table["groups"] for col in cols]


def _blocks(rows):
    """Row spans of consecutive equal first cells: [(start, length)]."""
    spans, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i][0] != rows[start][0]:
            spans.append((start, i - start))
            start = i
    return spans


def _grouped(table):
    return any(label for label, _ in table["groups"])


def _to_csv(table):
    with (OUT / f"{table['name']}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if _grouped(table):
            writer.writerow([""] * len(table["stub"])
                            + [label or "" for label, cols in table["groups"] for _ in cols])
        writer.writerow(table["stub"] + _columns(table))
        writer.writerows(table["rows"])


def _to_markdown(table):
    prefix = lambda label, col: f"{label}: {col}" if label else col
    header = table["stub"] + [prefix(label, col) for label, cols in table["groups"] for col in cols]
    lines = [f"**Table {table['number']}.** {table['caption']}", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    spans = dict(_blocks(table["rows"])) if len(table["stub"]) > 1 else {}
    for i, row in enumerate(table["rows"]):
        cells = list(row)
        if spans and i not in spans:
            cells[0] = ""  # repeated feature extraction label
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _to_html(table):
    n_stub = len(table["stub"])
    head = []
    if _grouped(table):
        first = "".join(
            f'<th rowspan="2" class="stub">{s}</th>' for s in table["stub"]
        ) + "".join(
            f'<th colspan="{len(cols)}" class="group">{label or ""}</th>'
            for label, cols in table["groups"]
        )
        head.extend(
            (
                f"<tr>{first}</tr>",
                "<tr>"
                + "".join(f"<th>{c}</th>" for c in _columns(table))
                + "</tr>",
            )
        )
    else:
        head.append("<tr>" + "".join(f'<th class="stub">{s}</th>' for s in table["stub"])
                    + "".join(f"<th>{c}</th>" for c in _columns(table)) + "</tr>")
    spans = dict(_blocks(table["rows"])) if n_stub > 1 else {}
    body = []
    for i, row in enumerate(table["rows"]):
        cells = []
        for j, cell in enumerate(row):
            if j == 0 and n_stub > 1:
                if i not in spans:
                    continue
                cells.append(f'<td rowspan="{spans[i]}" class="block">{cell}</td>')
            else:
                cells.append(f"<td>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table>\n<caption><b>Table {table["number"]}.</b> {table["caption"]}</caption>\n'
            f'<thead>\n{chr(10).join(head)}\n</thead>\n<tbody>\n{chr(10).join(body)}\n</tbody>\n</table>')


def _tex_escape(text):
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _to_latex(table):
    n_stub, columns = len(table["stub"]), _columns(table)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{{_tex_escape(table['caption'])}}}", rf"\label{{tab:{table['name']}}}",
             rf"\begin{{tabular}}{{{'l' * n_stub}{'c' * len(columns)}}}", r"\toprule"]
    if _grouped(table):
        cells, rules, col = [""] * n_stub, [], n_stub + 1
        for label, cols in table["groups"]:
            cells.append(rf"\multicolumn{{{len(cols)}}}{{c}}{{\textbf{{{_tex_escape(label or '')}}}}}")
            rules.append(rf"\cmidrule(lr){{{col}-{col + len(cols) - 1}}}")
            col += len(cols)
        lines += [" & ".join(cells) + r" \\", " ".join(rules)]
    lines.extend(
        (
            " & ".join(
                rf"\textbf{{{_tex_escape(c)}}}"
                for c in table["stub"] + columns
            )
            + r" \\",
            r"\midrule",
        )
    )
    spans = dict(_blocks(table["rows"])) if n_stub > 1 else {}
    for i, row in enumerate(table["rows"]):
        cells = [_tex_escape(c) for c in row]
        if spans:
            if i in spans:
                if i:
                    lines.append(r"\midrule")
                cells[0] = rf"\multirow{{{spans[i]}}}{{*}}{{{cells[0]}}}"
            else:
                cells[0] = ""
        if not spans and row[0] == "All data":
            lines.append(r"\midrule")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


HTML_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ablation tables</title>
<style>
  body {{ font-family: "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif; color: #111;
         background: #fff; margin: 0; padding: 32px 16px; }}
  main {{ max-width: 1100px; margin: 0 auto; display: grid; gap: 40px; }}
  p.note {{ margin: 0; color: #444; font-size: 0.9rem; }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; font-variant-numeric: tabular-nums; }}
  caption {{ caption-side: top; padding-bottom: 10px; font-size: 0.95rem; }}
  thead {{ border-top: 1.5px solid #111; border-bottom: 1px solid #111; }}
  tbody {{ border-bottom: 1.5px solid #111; }}
  th, td {{ padding: 6px 10px; text-align: center; white-space: nowrap; }}
  th {{ font-weight: 700; }}
  th.group {{ border-bottom: 1px solid #111; }}
  tbody tr + tr td {{ border-top: 1px solid #c8c8c8; }}
  td.block {{ border-right: 0; vertical-align: middle; }}
</style></head>
<body><main>
<p class="note">{note}</p>
{tables}
</main></body></html>
"""


def build_tables(datasets):
    collected = load_results(datasets)
    labels = [DATASETS[d] for d in datasets]
    row_keys = [*datasets, None]
    row_labels = [*(label.lower() for label in labels), "All data"]  # paper: "dataset 1" ... "All data"
    tables = []

    for number, selector in ((6, "LASSO"), (7, "LOG")):
        methods = [f"{e}+{selector}" for e in EXTRACTORS]
        tables.append(dict(
            number=number,
            name=f"table{number}_{selector.lower()}_secondary",
            caption=("The effect of secondary feature selection on average classification accuracy "
                     f"of various datasets (the {selector} method)."),
            stub=["Datasets"],
            groups=[("Without Secondary Feature Selection", methods),
                    ("With Secondary Feature Selection", methods)],
            rows=[[label, *(_mean(collected, datasets, e, f"{selector}_{mode}", d)
                            for mode in ("without", "with") for e in EXTRACTORS)]
                  for label, d in zip(row_labels, row_keys)],
        ))

    selectors = (("F-score", "F-score"), ("GA", "GA"), ("BPSO", "BPSO"),
                 ("LASSO", "LASSO_with"), ("LOG", "LOG_with"))
    tables.append(dict(
        number=8,
        name="table8_feature_selection",
        caption="The effect of different feature selection methods on average classification accuracy "
                "of various datasets.",
        stub=["Feature Extraction", "Feature Selection"],
        groups=[(None, [*labels, "All Data"])],
        rows=[[e, selector, *(_mean(collected, datasets, e, metric, d) for d in row_keys)]
              for e in EXTRACTORS for selector, metric in selectors],
    ))

    classifiers = (("SVM", "LOG_SVM"), ("BLDA", "LOG_BLDA"), ("FLDA", "LOG_with"))
    tables.append(dict(
        number=9,
        name="table9_classifiers",
        caption="The effect of different classifiers on average classification accuracy of various datasets.",
        stub=["Datasets"],
        groups=[(f"{e}+LOG", [c for c, _ in classifiers]) for e in EXTRACTORS],
        rows=[[label, *(_mean(collected, datasets, e, metric, d)
                        for e in EXTRACTORS for _, metric in classifiers)]
              for label, d in zip(row_labels, row_keys)],
    ))

    OUT.mkdir(parents=True, exist_ok=True)
    note = ("Average classification accuracy (%). \"All data\" pools every subject and Dataset 2 class pair, "
            "as in the paper's Tables 6-9.")
    for table in tables:
        _to_csv(table)
    markdown = note + "\n\n" + "\n\n".join(_to_markdown(t) for t in tables) + "\n"
    (OUT / "ablation_tables.md").write_text(markdown, encoding="utf-8")
    (OUT / "ablation_tables.html").write_text(
        HTML_PAGE.format(note=note, tables="\n".join(f'<div class="wrap">{_to_html(t)}</div>' for t in tables)),
        encoding="utf-8")
    (OUT / "ablation_tables.tex").write_text(
        "% Requires \\usepackage{booktabs} and \\usepackage{multirow}\n\n"
        + "\n\n".join(_to_latex(t) for t in tables) + "\n", encoding="utf-8")
    print(markdown)
    print(f"Saved tables to {OUT} (.md, .html, .tex and one .csv per table)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--jobs", type=int, default=-2, help="Parallel workers (joblib; default all but one core).")
    parser.add_argument("--tables-only", action="store_true", help="Only rebuild tables from the cache.")
    args = parser.parse_args()

    if not args.tables_only:
        tasks = [(d, trial, e) for d in args.dataset for trial in iter_dataset_trials(d) for e in EXTRACTORS]
        todo = [t for t in tasks if not _cache_path(*t).exists()]
        print(f"{len(tasks)} subject x extractor tasks, {len(todo)} to compute", flush=True)
        Parallel(n_jobs=args.jobs, verbose=5)(delayed(_run_task)(*t) for t in todo)
    build_tables(args.dataset)


if __name__ == "__main__":
    main()
