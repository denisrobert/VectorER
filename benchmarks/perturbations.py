"""Clerical and transmission perturbation types for the confusion-matrix
benchmark.

Each perturbation is a *true-match* variant: it mutates a copy of a base
record while keeping the same underlying person, so a query generated this way
is always labelled ``match`` (label 1).  The perturbations simulate the kinds
of errors a clerk (or a data-entry system) introduces:

* ``identity``            -- baseline, no change.
* ``first_initial``       -- the first name is initialized ("Evelyn" -> "E.").
* ``typo_names``          -- substitution / insertion / deletion / transposition
                            (metathesis) typos in the first name, last name, or
                            both.
* ``typo_address_email``  -- the same typo operations in the address and/or
                            email (email mutations stay in the local part so
                            the domain stays realistic).
* ``missing_address_email`` -- one or both of address / email are dropped
                            (``None``).
* ``non_standardized_address`` -- address rewritten in realistic, still-valid
                            forms: street-type abbreviations vs expansions
                            (``St`` <-> ``Street``, including wayfare kinds
                            like ``Blvd``/``Boulevard``, ``Cres``/``Crescent``,
                            ``Pkwy``/``Parkway``), inclusion of civic/unit
                            reference forms (``-5`` suffixes, ``Apt 5``,
                            ``Unit 5``, ``#5``), field-order variants (civic
                            number after the street, postal code relocation,
                            province/city swaps), and PO-box / RR substitutions.
* ``address_change``      -- the person genuinely **moved within the same city
                            and postal-code prefix** (the most frequent move).
                            The street name, civic number, and/or street type
                            change to a different street in the same city; the
                            city, province, and forward-sortation-area prefix of
                            the postal code are preserved (only the trailing
                            portion of the postal code may change).  This tests
                            resilience to a common person-ER reality: an old
                            address no longer matches at all.
* ``serialization``       -- generic serialization/transmission errors that are
                            not simple typos: garbled whitespace, case flips,
                            field reordering, separator corruption, stray
                            characters, truncation, abbreviation expansion.

All functions take ``(record, rng)`` and return a *copy*; ``rng`` is a
``random.Random`` seeded by the caller, so runs are reproducible.  The different
mysterious-body types are exposed through :data:`PERTURBATION_TYPES`.
"""

from __future__ import annotations

import random
from typing import Callable

ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _copy(record: dict) -> dict:
    return dict(record)


# ---------------------------------------------------------------------------
# Typo operations (character-level editing)
# ---------------------------------------------------------------------------


def typo_once(value: str, rng: random.Random, n: int = 1) -> str:
    """Apply ``n`` random character-level edits to a string: substitution,
    insertion (addition), deletion, or transposition (metathesis)."""
    out = value
    for _ in range(n):
        if not out:
            break
        op = rng.randint(0, 3)
        chars = list(out)
        if op == 0:  # substitution
            i = rng.randrange(len(chars))
            chars[i] = rng.choice(ALPHA)
        elif op == 1:  # insertion (addition)
            i = rng.randrange(len(chars) + 1)
            chars.insert(i, rng.choice(ALPHA))
        elif op == 2:  # deletion
            i = rng.randrange(len(chars))
            del chars[i]
        else:  # transposition (metathesis)
            if len(chars) < 2:
                continue
            i = rng.randrange(len(chars) - 1)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        out = "".join(chars)
    return out


# ---------------------------------------------------------------------------
# Perturbation types
# ---------------------------------------------------------------------------


def perturb_identity(record: dict, rng: random.Random) -> dict:
    """Baseline: return the record unchanged."""
    del rng
    return _copy(record)


def perturb_first_initial(record: dict, rng: random.Random) -> dict:
    """Initialize the first name (``"Evelyn"`` -> ``"E."``)."""
    del rng
    out = _copy(record)
    first = out.get("first_name")
    if first:
        out["first_name"] = first[0] + "."
    return out


def perturb_typo_names(record: dict, rng: random.Random, edits: int = 1) -> dict:
    """Typos in first, last, or both; each applied typo is substitution,
    insertion, deletion, or transposition."""
    out = _copy(record)
    choice = rng.randint(0, 2)  # 0=first, 1=last, 2=both
    targets = []
    if choice in (0, 2):
        targets.append("first_name")
    if choice in (1, 2):
        targets.append("last_name")
    for field in targets:
        value = out.get(field)
        if value and len(value) > 1:
            out[field] = typo_once(value, rng, n=max(1, int(round(rng.random() * edits + 0.5))))
        elif value:
            out[field] = typo_once(value, rng, n=1)
    return out


def perturb_typo_address_email(record: dict, rng: random.Random, edits: int = 1) -> dict:
    """Typos in the address and/or email (kept in the email local part)."""
    out = _copy(record)
    ar = rng.random()
    if out.get("address") and ar < 0.8:
        out["address"] = typo_once(out["address"], rng, n=max(1, int(round(rng.random() * edits + 1))))
    if out.get("email") and (ar >= 0.4 or not out.get("address")):
        local, _, domain = out["email"].partition("@")
        if local:
            local = typo_once(local, rng, n=max(1, int(round(rng.random() * edits + 1))))
            out["email"] = f"{local}@{domain}"
    return out


def perturb_missing_address_email(record: dict, rng: random.Random) -> dict:
    """Drop the address and/or the email (None)."""
    out = _copy(record)
    r = rng.random()
    if r < 0.45:
        out["address"] = None  # missing address
    elif r < 0.9:
        out["email"] = None  # missing email
    else:
        out["address"] = None
        out["email"] = None  # both missing
    return out


def _serialize_garbled(record: dict, rng: random.Random) -> dict:
    out = _copy(record)
    mode = rng.randint(0, 7)

    def pick_field() -> str:
        fields = [f for f in ("first_name", "last_name", "address", "email") if out.get(f)]
        return rng.choice(fields) if fields else "address"

    if mode == 0:  # whitespace garbling: collapse/expand, tab/space mixes
        field = pick_field()
        v = out[field]
        if rng.random() < 0.5:
            out[field] = "  ".join(v.split())
        else:
            out[field] = v.replace(" ", "\t")
    elif mode == 1:  # case corruption (all-caps or alternating)
        field = pick_field()
        v = out[field]
        if rng.random() < 0.5:
            out[field] = v.upper()
        else:
            out[field] = "".join(
                ch.upper() if i % 2 else ch.lower() for i, ch in enumerate(v)
            )
    elif mode == 2:  # field reordering in the address (city/province swap etc.)
        if out.get("address"):
            addr = out["address"]
            if ", " in addr:
                parts = addr.split(", ")
                if len(parts) >= 3 and rng.random() < 0.5:
                    city, prov = parts[-2], parts[-1]
                    parts[-2], parts[-1] = prov, city
                out["address"] = ", ".join(parts)
    elif mode == 3:  # stray characters / delimiters injected
        field = pick_field()
        v = str(out.get(field) or "")
        i = rng.randrange(len(v) + 1) if v else 0
        out[field] = v[:i] + rng.choice(["#", "*", "!", "  ", "\\", "/"]) + v[i:]
    elif mode == 4:  # truncation of a long field
        field = pick_field()
        v = str(out.get(field) or "")
        if len(v) > 4:
            out[field] = v[: rng.randint(len(v) // 2, len(v) - 1)]
    elif mode == 5:  # abbreviation expansion (St -> Street etc.)
        if out.get("address"):
            addr = out["address"]
            for old, new in [("St", "Street"), ("Ave", "Avenue"), ("Rd", "Road"),
                             ("Blvd", "Boulevard"), ("Cres", "Crescent"),
                             ("Dr", "Drive"), ("Ct", "Court")]:
                if old in addr and rng.random() < 0.6:
                    addr = addr.replace(old, new)
                    break
            out["address"] = addr
    elif mode == 6:  # word-order shuffle inside a name or street
        field = rng.choice(["first_name", "last_name"]) if out.get("first_name") else "last_name"
        v = out.get(field) or ""
        if len(v) > 1:
            chars = list(v)
            # swap two adjacent words/letters that look like a transmission slip
            i = rng.randrange(len(chars) - 1)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            out[field] = "".join(chars)
    else:  # postal-code separator corruption / whitespace
        if out.get("address"):
            addr = out["address"]
            if " " in addr:
                i = [idx for idx, c in enumerate(addr) if c == " "]
                j = rng.choice(i)
                addr = addr[:j] + rng.choice(["-", "_", "", "\t"]) + addr[j + 1:]
            out["address"] = addr
    return out


def perturb_serialization(record: dict, rng: random.Random) -> dict:
    """Simulate generic serialization / transmission errors on the record.

    Several independent errors may compound, mimicking a corrupt serialized
    payload rather than a single clerical typo.
    """
    out = _copy(record)
    for _ in range(rng.randint(1, 3)):
        out = _serialize_garbled(out, rng)
    return out


# ---------------------------------------------------------------------------
# Non-standardized addresses
# ---------------------------------------------------------------------------

#: Street-type "wayfare" abbreviation <-> expansion pairs and extra kinds.
STREET_KIND_PAIRS = [
    ("St", "Street"), ("Ave", "Avenue"), ("Rd", "Road"), ("Blvd", "Boulevard"),
    ("Cres", "Crescent"), ("Dr", "Drive"), ("Ct", "Court"), ("Pkwy", "Parkway"),
    ("Way", "Way"), ("Ln", "Lane"), ("Hwy", "Highway"), ("N", "North"),
    ("S", "South"), ("E", "East"), ("W", "West"),
]
#: Unit/civic reference phrasing variants.
UNIT_FORMS = ["Apt {n}", "Unit {n}", "Suite {n}", "#{n}", "Ste {n}", "Rm {n}"]


def _parse_address(addr: str) -> dict | None:
    """Parse the census-style ``"123 Main St, City, Province PC"``.

    ``province`` is the province/territory name; ``postal`` is the full
    ``"A1A 1A1"`` (or ``None``).  Handles a trailing two-part postal code.
    """
    if not addr or ", " not in addr:
        return None
    parts = addr.split(", ")
    if len(parts) != 3:
        return None
    street_part = parts[0]
    pieces = street_part.split()
    if len(pieces) < 2:
        return None
    # try civic number prefix then the rest as street
    if pieces[0].isdigit() or (pieces[0][:-1].isdigit() and pieces[0][-1] in "-/"):
        civic = pieces.pop(0)
        street = " ".join(pieces)
    else:
        civic = None
        street = street_part
    city = parts[1]
    # last field: "Province A1A 1A1" (postal is the trailing two tokens).
    tail = parts[2]
    toks = tail.split()
    if len(toks) >= 3:
        postal = " ".join(toks[-2:])
        province = " ".join(toks[:-2])
    else:
        postal = None
        province = tail
    return {"civic": civic, "street": street, "city": city,
            "province": province, "postal": postal}


def _swap_street_kind(street: str | None, rng: random.Random) -> str:
    """Expand or abbreviate a street-type suffix (St<->Street, etc.)."""
    if not street:
        return street or ""
    words = street.split()
    for i in range(len(words) - 1, -1, -1):
        for abbr, full in STREET_KIND_PAIRS:
            if words[i].lower() == abbr.lower():
                words[i] = full if words[i][0].isupper() else full.lower()
                return " ".join(words)
            if words[i].lower() == full.lower():
                # reintroduce an abbreviation for multi-word kinds only
                if len(full) > 4:
                    words[i] = abbr
                    return " ".join(words)
    # no recognizable type -> append a canonical wayfare word
    extra = rng.choice(["St", "Blvd", "Cres"])
    return f"{street} {extra}"


def perturb_non_standardized_address(record: dict, rng: random.Random) -> dict:
    """Rewrite the address into a realistically non-standardized variant.

    Keeps the *same* underlying location but changes how it is written:
    street-type abbreviations/expansions, civic/unit reference phrasing, field
    order, and occasionally a PO-box/rural-route substitution.  Applicable only
    when the record has an address.
    """
    out = _copy(record)
    addr = out.get("address")
    if not addr:
        return out
    parsed = _parse_address(str(addr))
    if parsed is None:
        # Fallback: apply a couple of generic normalization twists.
        out["address"] = _swap_street_kind(str(addr), rng)
        return out

    n_variations = rng.randint(1, 3)
    for _ in range(n_variations):
        mode = rng.randint(0, 5)
        if mode == 0:  # expand/abbreviate a street-type suffix (wayfare kinds)
            parsed["street"] = _swap_street_kind(parsed["street"], rng)
        elif mode == 1:  # unit/civic reference forms (only when street is a real street)
            unit_n = rng.randint(1, 120)
            form = rng.choice(UNIT_FORMS).format(n=unit_n)
            is_rural = (parsed.get("street") or "").upper().startswith(("RR ", "PO ", "SITE", "COMP"))
            if parsed.get("civic"):
                # "1425-51" civic-unit form or a "#51" prefix
                parsed["civic"] = f"{parsed['civic']}-{unit_n}" if rng.random() < 0.5 else parsed["civic"]
            elif parsed.get("street") and not is_rural and not parsed.get("POBox"):
                parsed["street"] = f"{form} {parsed['street']}"
        elif mode == 2:  # field order: postal relocation or street-word shuffle
            if rng.random() < 0.35 and parsed.get("postal"):
                parsed["postal_relocated"] = True
            if parsed.get("street"):
                street_words = parsed["street"].split()
                if len(street_words) > 2 and rng.random() < 0.5:
                    i = rng.randrange(len(street_words) - 1)
                    street_words[i], street_words[i+1] = street_words[i+1], street_words[i]
                    parsed["street"] = " ".join(street_words)
        elif mode == 3:  # street-type abbreviation when none present
            parsed["street"] = _swap_street_kind(parsed["street"], rng)
        elif mode == 4:  # punctuation: comma-less or extra spacing
            pass  # handled in the serializer rebuild
        else:  # PO box / rural-route substitute (rare)
            if rng.random() < 0.3:
                parsed["POBox"] = f"PO Box {rng.randint(1, 9999)}, {parsed['city']}"
                parsed["street"] = None
                parsed["civic"] = None  # no street number
            else:
                parsed["street"] = f"RR {rng.randint(1, 9)} Site {rng.randint(1, 99)} Comp {rng.randint(1, 99)}"
                parsed["civic"] = None  # rural routes have no civic number

    if parsed.get("POBox"):
        out["address"] = parsed["POBox"] + f", {parsed['province']} {parsed['postal']}"
        return out

    street_parts = ([parsed["civic"]] if parsed.get("civic") else []) + ([parsed["street"]] if parsed.get("street") else [])
    street = " ".join(p for p in street_parts if p)

    # Assemble in one of several realistic orderings.
    if parsed.get("postal_relocated"):
        orderings = [2]  # forced: postal code pulled next to the street
    else:
        orderings = [0, 1]  # standard or city-prefixed
    ordering = rng.choice(orderings)
    if ordering == 0:
        out["address"] = f"{street}, {parsed['city']}, {parsed['province']} {parsed['postal']}"
    elif ordering == 1:
        out["address"] = f"{parsed['city']}: {street}; {parsed['province']} {parsed['postal']}"
    else:
        out["address"] = f"{street} {parsed['postal']}, {parsed['city']}, {parsed['province']}"
    return out


# Canadian street vocabulary (mirrors the census generator's).
_MOVE_STREETS = ["Main", "Queen", "King", "Elm", "Maple", "Oak", "Cedar", "Birch",
                 "Park", "Church", "Victoria", "Wellington", "Bay", "Lake", "Hill",
                 "River", "Douglas", "Albert", "Chestnut", "Spruce", "Willow",
                 "McLean", "Sherbrooke", "Colborne", "Front", "Market", "Union"]
_MOVE_STREET_KINDS = ["St", "Ave", "Rd", "Blvd", "Cres", "Dr", "Way", "Ln", "Ct", "Pl"]


def perturb_address_change(record: dict, rng: random.Random) -> dict:
    """Simulate a **move within the same city and postal-code prefix**.

    The person's new address is on a different street (different name, number,
    and/or street type) but stays in:
    - the same city,
    - the same province,
    - the same forward-sortation-area prefix of the postal code (the first
      three characters, e.g. ``M5V``), which encodes the city/urban zone.

    Only the trailing local-delivery-unit portion of the postal code may change
    (consistent with moving a few blocks).  The *old* address is thus entirely
    different, testing whether the engine can still link the person via the
    remaining fields (name, DOB, email).
    """
    out = _copy(record)
    addr = out.get("address")
    if not addr:
        return out
    parsed = _parse_address(str(addr))
    if parsed is None:
        return out  # cannot relocate reliably; leave unchanged (rare)

    # Preserve the city/province forward-sortation-area prefix.
    city = parsed["city"]
    province = parsed["province"]
    old_fsa = (parsed.get("postal") or "A1A").split()[0][:3]

    # New street: different name/kind/number.  Reuse a street different from the
    # current one when possible, and vary the civic number.
    current_street = (parsed.get("street") or "").lower()
    pool = [s for s in _MOVE_STREETS if s.lower() not in current_street] or _MOVE_STREETS
    new_street = rng.choice(pool)
    new_kind = rng.choice(_MOVE_STREET_KINDS)
    new_civic = str(rng.randint(1, 9999))

    # Reuse the FSA prefix and regenerate a plausible LDU (3rd char + 4/5/6th).
    fsa_letter_2 = old_fsa[1] if len(old_fsa) > 1 else rng.choice("ABCDEFGHIJKLMNPRSTUVWXYZ")
    fsa_digit = old_fsa[2] if len(old_fsa) > 2 else "1"
    fsa_extra = rng.choice("ABCDEFGHIJKLMNPRSTUVWXYZ")
    ldu_digit1 = "1" if rng.random() < 0.95 else "0"
    ldu_letter = rng.choice("ABCDEFGHIJKLMNPRSTUVWXYZ")
    ldu_digit2 = str(rng.randint(1, 9))
    new_pc = f"{old_fsa[0]}{fsa_letter_2}{fsa_digit} {ldu_digit1}{ldu_letter}{ldu_digit2}"

    out["address"] = f"{new_civic} {new_street} {new_kind}, {city}, {province} {new_pc}"
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PERTURBATION_TYPES: dict[str, Callable[[dict, random.Random], dict]] = {
    "identity": perturb_identity,
    "first_initial": perturb_first_initial,
    "typo_names": perturb_typo_names,
    "typo_address_email": perturb_typo_address_email,
    "missing_address_email": perturb_missing_address_email,
    "non_standardized_address": perturb_non_standardized_address,
    "address_change": perturb_address_change,
    "serialization": perturb_serialization,
}


def apply_perturbation(kind: str, record: dict, rng: random.Random) -> dict:
    """Apply the named perturbation to a copy of ``record``."""
    if kind not in PERTURBATION_TYPES:
        raise KeyError(f"unknown perturbation {kind!r}; available: {sorted(PERTURBATION_TYPES)}")
    return PERTURBATION_TYPES[kind](record, rng)