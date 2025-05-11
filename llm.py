# llm.py

import sys
import glob
import os

# read through all the files in the folder and its subfolders, and save the text of each one which is *.py, *.rs, or *.toml to a single file called "code.md"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python llm.py <folder>")
        sys.exit(1)
    folder = sys.argv[1]

    files = glob.glob(os.path.join(folder, "**", "*"), recursive=True)
    files = [
        f
        for f in files
        if f.endswith(".py") or f.endswith(".rs") or f.endswith(".toml")
    ]

    with open("code.md", "w") as f:
        for file in files:
            f.write(f"### {file}\n\n```\n")
            with open(file, "r") as f2:
                f.write(f2.read())
            f.write("\n```\n\n\n")
