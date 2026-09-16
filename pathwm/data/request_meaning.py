"""Authored request contrasts; controlled composition, not a language benchmark."""

# Same prefix and exact UTF-8 length for opposite requested answer formats.
PAYLOADS = (
    ("nur die allererste Farbe", "alle Farben nacheinander"),
    ("den zuerst gezeigten Farbton", "beide Farbtöne nacheinander"),
)
TEMPLATES = {
    "calibration": (
        "Nenne {}.",
        "Zeige {}.",
        "Gib {} an.",
        "Schreibe {} auf.",
        "Bitte nenne {}.",
        "Als Antwort nenne {}.",
    ),
    "validation": ("Bitte zeige {}.", "Gib mir {} an."),
    "test": (
        "Schreibe mir {} auf.",
        "Bitte gib {} an.",
        "Als Antwort zeige {}.",
        "Nenne mir bitte {}.",
    ),
}


def request_corpus():
    rows = []
    for split, templates in TEMPLATES.items():
        for family, template in enumerate(templates):
            for wording, payloads in enumerate(PAYLOADS):
                pair = f"{split}/{family}/{wording}"
                for label, payload in enumerate(payloads):
                    rows.append(
                        dict(
                            id=f"{pair}/{label}",
                            pair=pair,
                            family=f"{split}/{family}",
                            split=split,
                            style="direct",
                            question=template.format(payload),
                            label=label,
                            format=("first", "sequence")[label],
                        )
                    )
    # Both strings have exactly the same byte multiset; word order changes meaning.
    for family, template in enumerate(TEMPLATES["test"][:2]):
        for label in range(2):
            phrases = ("die erste Farbe", "beide Farben der Reihe nach")
            phrase = phrases[label] + ", nicht " + phrases[1 - label]
            pair = f"order/{family}"
            rows.append(
                dict(
                    id=f"{pair}/{label}",
                    pair=pair,
                    family=pair,
                    split="test",
                    style="order",
                    question=template.format(phrase),
                    label=label,
                    format=("first", "sequence")[label],
                )
            )
    return rows


def apply_request(record, request):
    """Keep evidence/splits; labels construct targets only, never network metadata."""
    if record["case"] != "VID.order":
        raise ValueError("Request meanings currently require VID.order")
    choices = record["choices"]
    if request["format"] == "sequence":
        choices = [" ".join((choices[j], choices[1 - j])) for j in range(2)]
    return record | dict(
        id=f"{record['id']}/{request['id']}",
        pair=f"{record['pair']}/{request['id']}",
        question=request["question"],
        choices=choices,
        format=request["format"],
        request_id=request["id"],
    )
