"""Decode sealed, hash-verifiable raw DDS archives without touching the robot.
Run in the pinned recorder runtime: python export_cdr.py INPUT.jsonl.gz OUTPUT.jsonl.gz
Input is DDS CDR bytes, never executable pickle. Unknown/invalid rows fail visibly.
"""

import base64
import gzip
import json
import sys
from recorder import TOPICS, plain
from unitree_sdk2py.idl.unitree_hg.msg import dds_


def decode(row):
    if "cdr_base64" in row:
        msg = getattr(dds_, TOPICS[row["topic"]]).deserialize(
            base64.b64decode(row.pop("cdr_base64"), validate=True)
        )
        row["message"] = plain(msg)
        row["derived_from_cdr"] = True
    return row


if __name__ == "__main__":
    with (
        gzip.open(sys.argv[1], "rt") as src,
        gzip.open(sys.argv[2], "wt", compresslevel=1) as dest,
    ):
        for line in src:
            dest.write(
                json.dumps(
                    decode(json.loads(line)), separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
