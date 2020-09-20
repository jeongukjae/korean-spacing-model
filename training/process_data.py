from argparse import ArgumentParser

parser = ArgumentParser(description="Remove empty line and title line in `in-file` and write to the `out-file`.")
parser.add_argument("--in-file", type=str, required=True)
parser.add_argument("--out-file", type=str, required=True)

args = parser.parse_args()

with open(args.in_file) as in_file, open(args.out_file, "w") as out_file:
    for index, line in enumerate(in_file):
        line = line.strip()
        if line == "":
            continue
        if line.startswith("= ") and line.endswith(" ="):
            continue

        print(line, file=out_file)

        if index % 10 == 0:
            print(f"processed {index} lines")
