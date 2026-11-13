"""Quick start: migrate two stored templates, run the first login, re-enrol, then a daily login.

    python examples/quickstart.py --template u001.iso --probe u001_probe.png --today u001_today.png
"""
import argparse

from fpmig import MigrationClient

ap = argparse.ArgumentParser()
ap.add_argument("--template", required=True, help="ISO/IEC 19794-2 record stored by the legacy system")
ap.add_argument("--probe", required=True, help="image captured at the user's first login")
ap.add_argument("--today", required=True, help="image captured at a later login")
ap.add_argument("--models", help="model folder or zip (default: download the release)")
a = ap.parse_args()

client = MigrationClient(a.models)
migration_template = client.migrate_template(open(a.template, "rb").read())       # offline, once per template

first = client.first_login(a.probe, migration_template)                            # once per user
print("first login:", first.as_dict())
if not first.match:
    raise SystemExit("not accepted: retry, or re-enrol the user under supervision")

enrolled = first.image_embedding                                                   # store; drop the migration template
daily = client.verify(a.today, enrolled)                                           # every later login
print("daily login:", daily.as_dict())
