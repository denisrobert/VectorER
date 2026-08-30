"""Example: record linkage across two databases (merger / cross-enterprise).

Two independently-managed databases with **different schemas** that overlap on
the compared fields are *linked*, not merged: the output is a table of
``(a_id, b_id)`` link edges with the FS posterior and decision band.  Each
database keeps its own schema, ids and data.

Run from the project root:

    python examples/link_two_databases.py [--mode directed|symmetric] [--tau 0.7]
"""

from __future__ import annotations

import argparse

from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.link import FieldMap, RecordLinker
from vectorer.scoring import FellegiSunterScorer


def build_company_db_a():
    """Merging party A: customers with a CRM schema."""
    return [
        {"cust_id": "c1", "name": "john smith", "birth_date": "1985-06-15", "email_c": "john@a.com"},
        {"cust_id": "c2", "name": "mary jones", "birth_date": "1990-11-03", "email_c": None},
        {"cust_id": "c3", "name": "robert martin", "birth_date": "1978-02-28", "email_c": "rob@a.com"},
        {"cust_id": "c4", "name": "linda brown", "birth_date": "1988-07-19", "email_c": "linda@a.com"},
    ]


def build_partner_db_b():
    """Collaboration party B: an ERP schema with different column names."""
    return [
        {"partner_id": "p1", "legal_name": "jon smith", "dob": "1985-06-15", "email_p": "jsmith@b.com"},
        {"partner_id": "p2", "legal_name": "robert martinez", "dob": None, "email_p": None},
        {"partner_id": "p3", "legal_name": "zoe kwan", "dob": "1999-01-01", "email_p": "zoe@b.com"},
        {"partner_id": "p4", "legal_name": "linda brown", "dob": "1988-07-19", "email_p": "linda@b.com"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-database record linkage example")
    parser.add_argument("--mode", choices=["directed", "symmetric"], default="directed")
    parser.add_argument("--tau", type=float, default=0.70)
    args = parser.parse_args()

    comparisons = [
        make_comparison("jaro_winkler_at_thresholds", col_name="name",
                        score_threshold_or_thresholds=[0.9, 0.8, 0.7]),
        make_comparison("date_of_birth_comparison", col_name="dob"),
        make_comparison("email_comparison", col_name="email"),
    ]
    scorer = FellegiSunterScorer.from_comparisons(comparisons, prior=1e-2, threshold=args.tau)

    linker = RecordLinker(
        embedder=CharacterHashingEmbedding(dimension=384),
        scorer=scorer,
        field_maps={
            # canonical: source column in each DB's own schema
            "A": FieldMap({"name": "name", "dob": "birth_date", "email": "email_c"},
                          id_column="cust_id"),
            "B": FieldMap({"name": "legal_name", "dob": "dob", "email": "email_p"},
                          id_column="partner_id"),
        },
        k=4,
        tau=args.tau,
        possible_low=0.5,  # three-tier: match / possible / non-match
    )

    a = build_company_db_a()
    b = build_partner_db_b()
    tbl = linker.link(a, b, mode=args.mode)

    print(f"Linked {len(a)} A-records x {len(b)} B-records "
          f"({args.mode} mode): {tbl.n_matches} links, "
          f"{tbl.n_possible_matches} possible")
    print(f"{'A':8s} {'B':8s} {'p(match)':>9s} {'decision':14s}")
    for e in sorted(tbl.edges, key=lambda e: -e.probability):
        print(f"{str(e.a_id):8s} {str(e.b_id):8s} {e.probability:9.3f} {e.decision:14s}")

    print("\nMatched pairs:", tbl.as_pairs())


if __name__ == "__main__":
    main()