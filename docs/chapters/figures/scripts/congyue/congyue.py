import os


eventid_file = os.path.join(os.path.dirname(__file__), "events.txt")

with open(eventid_file, 'r') as f:
    eventids = f.read().splitlines()


