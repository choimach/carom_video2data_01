"""The naming rule has to keep beating the ones it replaced.

A rule lives or dies on `data/labels.json` - the plays the player named himself
while watching the video. He is learning the game as well, so nothing he says is
taken as law; it is a hypothesis, and this is where hypotheses are scored.

The test exists because prose does not stop anyone. Three times a rule was
invented here that read well and scored worse, and the record that would have
caught it sat unread in a document. A number that fails the build does not sit
unread.
"""

import pytest

from tools.rules_check import RULES, TURNS, by_family, labelled


@pytest.fixture(scope="module")
def turns():
    rows = [(route, play) for route, play in labelled() if route in TURNS]
    if len(rows) < 20:
        pytest.skip("the dataset has not been exported with the labelled matches")
    return rows


def score(rule, rows):
    return sum(1 for route, play in rows if rule(play) == route) / len(rows)


def test_the_rule_in_place_names_four_in_five(turns):
    assert score(by_family, turns) >= 0.80


def test_it_beats_every_rule_it_replaced(turns):
    best = score(by_family, turns)
    for name, rule in RULES:
        if rule is by_family:
            continue
        assert score(rule, turns) < best, f"{name} now scores as well - re-open the question"
