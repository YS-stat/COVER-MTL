"""Prepare the public GTEx v8 central-nervous-system data for analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_GENES = ("JAM2", "SH2D2A")


def tissue_name(path: Path) -> str:
    prefix = "gene_tpm_2017-06-05_v8_"
    name = path.name.removeprefix(prefix).removesuffix(".gct.gz")
    return name


def donor_id(sample_id: str) -> str:
    match = re.match(r"^(GTEX-[^-]+)-", sample_id)
    if match is None:
        raise ValueError(f"Unexpected GTEx sample identifier: {sample_id!r}")
    return match.group(1)


def read_module_genes(path: Path) -> list[str]:
    tables = pd.read_html(path)
    if len(tables) != 1 or tables[0].shape[1] < 2:
        raise ValueError("The MODULE 137 page has an unexpected table structure.")
    table = tables[0].copy()
    table.columns = ["entrez", "symbol", "p_value", "description"]
    table = table.iloc[1:].copy()
    symbols = (
        table["symbol"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    )
    return sorted(set(symbols.dropna().tolist()))


def read_selected_expression(path: Path, genes: set[str]) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep="\t", skiprows=2, chunksize=5000):
        keep = chunk["Description"].astype(str).isin(genes)
        if keep.any():
            selected.append(
                chunk.loc[keep].drop(columns=["id", "Name"]).rename(
                    columns={"Description": "gene"}
                )
            )
    if not selected:
        raise ValueError(f"No requested genes were found in {path}.")
    expression = pd.concat(selected, ignore_index=True)
    numeric_columns = [column for column in expression.columns if column != "gene"]
    expression[numeric_columns] = expression[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    if expression["gene"].duplicated().any():
        expression = expression.groupby("gene", as_index=False)[numeric_columns].mean()
    return expression.set_index("gene").T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared"))
    args = parser.parse_args()

    module_path = args.raw_dir / "module_137_full_genes.html"
    module_genes = read_module_genes(module_path)
    requested = set(module_genes) | set(TARGET_GENES)
    tissue_files = sorted(args.raw_dir.glob("gene_tpm_2017-06-05_v8_brain_*.gct.gz"))
    if len(tissue_files) != 11:
        raise ValueError(
            f"Expected 11 central-nervous-system tissue files, found {len(tissue_files)}."
        )

    frames: list[pd.DataFrame] = []
    tissue_summary: list[dict[str, object]] = []
    common_genes: set[str] | None = None
    for path in tissue_files:
        frame = read_selected_expression(path, requested)
        tissue = tissue_name(path)
        missing_targets = sorted(set(TARGET_GENES) - set(frame.columns))
        if missing_targets:
            raise ValueError(f"{tissue} is missing target genes {missing_targets}.")
        common_genes = set(frame.columns) if common_genes is None else common_genes & set(frame.columns)
        frame.insert(0, "sample_id", frame.index.astype(str))
        frame.insert(1, "donor_id", [donor_id(value) for value in frame["sample_id"]])
        frame.insert(2, "tissue", tissue)
        frame = frame.reset_index(drop=True)
        frames.append(frame)
        tissue_summary.append(
            {
                "tissue": tissue,
                "samples": int(frame.shape[0]),
                "donors": int(frame["donor_id"].nunique()),
                "genes_available": int(frame.shape[1] - 3),
            }
        )

    if common_genes is None:
        raise RuntimeError("No expression matrices were loaded.")
    common = sorted(common_genes)
    ordered_columns = ["sample_id", "donor_id", "tissue", *common]
    combined = pd.concat([frame[ordered_columns] for frame in frames], ignore_index=True)
    expression_columns = common
    combined[expression_columns] = np.log2(combined[expression_columns] + 1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output_dir / "gtex_v8_brain_module137.parquet", index=False)
    pd.DataFrame(tissue_summary).to_csv(
        args.output_dir / "tissue_summary.csv", index=False
    )
    pd.DataFrame({"gene": common}).to_csv(
        args.output_dir / "common_module_genes.csv", index=False
    )

    donor_counts = combined.groupby("donor_id")["tissue"].nunique()
    audit = {
        "module_gene_symbols": len(module_genes),
        "common_expression_genes": len(common),
        "predictor_genes": len(set(common) - set(TARGET_GENES)),
        "samples": int(combined.shape[0]),
        "unique_donors": int(combined["donor_id"].nunique()),
        "donors_in_multiple_tissues": int((donor_counts > 1).sum()),
        "maximum_tissues_per_donor": int(donor_counts.max()),
        "target_genes": list(TARGET_GENES),
        "transformation": "log2(TPM + 1)",
    }
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(pd.DataFrame(tissue_summary).to_string(index=False))


if __name__ == "__main__":
    main()
