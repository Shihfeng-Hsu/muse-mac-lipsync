#!/usr/bin/env python
# muse_tools.py - helpers for muse.sh (multi-avatar front end).
#
# Subcommands:
#   migrate <old_coords.pkl> <new_coords.pkl> <new_frames_dir>
#       Rewrite the frame paths inside a landmark cache to a new directory.
#       Uses each path's basename (assumes all frames live in one folder).
#   profile <name> <source_video> <coords.pkl> <out.json>
#       Write a small profile.json describing the avatar.
#
# All code and comments are pure ASCII by design.

import json
import os
import pickle
import sys
import time


def cmd_migrate(argv):
    if len(argv) != 3:
        print("usage: muse_tools.py migrate <old.pkl> <new.pkl> <new_frames_dir>")
        return 1
    old, new, new_frames = argv
    with open(old, "rb") as f:
        meta = pickle.load(f)
    frames = meta["frames"]
    new_paths = [os.path.join(new_frames, os.path.basename(p)) for p in frames]
    meta["frames"] = new_paths
    with open(new, "wb") as f:
        pickle.dump(meta, f)
    print("migrated %d frame paths -> %s" % (len(new_paths), new))
    return 0


def cmd_profile(argv):
    if len(argv) != 4:
        print("usage: muse_tools.py profile <name> <video> <coords.pkl> <out.json>")
        return 1
    name, video, coords, out = argv
    with open(coords, "rb") as f:
        meta = pickle.load(f)
    prof = {
        "name": name,
        "source_video": os.path.abspath(video),
        "source_size": os.path.getsize(video),
        "fps": float(meta["fps"]),
        "n_frames": len(meta["frames"]),
        "n_coords": len(meta["coords"]),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out, "w") as f:
        json.dump(prof, f, indent=2)
        f.write("\n")
    print(json.dumps(prof, indent=2))
    return 0


def main():
    if len(sys.argv) < 2:
        print("usage: muse_tools.py {migrate|profile} ...")
        return 1
    cmd = sys.argv[1]
    if cmd == "migrate":
        return cmd_migrate(sys.argv[2:])
    if cmd == "profile":
        return cmd_profile(sys.argv[2:])
    print("unknown subcommand: %s" % cmd)
    return 1


if __name__ == "__main__":
    sys.exit(main())