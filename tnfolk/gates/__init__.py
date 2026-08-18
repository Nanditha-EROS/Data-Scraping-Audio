"""Independently-testable pipeline gates.

Each gate is its own module exposing a single pure-ish entry function that
takes a Candidate (+ config/context) and returns a GateResult. Gates never let
one file's exception kill the batch -- callers catch, log with the recording id,
and mark REJECT(DOWNLOAD_FAILED) or similar before moving on.
"""
