"""只读取现有 .NET 文件的指令，检查指定方法及调用方；不加载或执行该程序。"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import dnfile
from dncil.cil.body.reader import read_method_body_from_bytes
from dncil.cil.error import MethodBodyFormatError
from dncil.clr.token import StringToken, Token


def inspect(path, pattern, depth, output):
    binary = dnfile.dnPE(str(path))
    owners = {}
    for row in binary.net.mdtables.TypeDef.rows:
        owner = f"{row.TypeNamespace}.{row.TypeName}"
        for item in row.MethodList:
            owners[item.row_index] = owner
    names = {
        i: f"{owners.get(i, '')}::{row.Name}"
        for i, row in enumerate(binary.net.mdtables.MethodDef.rows, 1)
    }

    def operand_text(value):
        if isinstance(value, StringToken):
            return str(binary.net.user_strings.get(value.rid))
        if isinstance(value, Token):
            if value.table == 6:
                return names[value.rid]
            table = binary.net.mdtables.tables.get(value.table)
            if table and 0 < value.rid <= len(table.rows):
                row = table.rows[value.rid - 1]
                return str(getattr(row, "Name", getattr(row, "TypeName", value)))
        return str(value)

    bodies, calls, errors = {}, {}, []
    for index, row in enumerate(binary.net.mdtables.MethodDef.rows, 1):
        if not row.Rva:
            continue
        try:
            body = read_method_body_from_bytes(binary.get_data(row.Rva))
        except MethodBodyFormatError as error:
            errors.append({"method": names[index], "error": str(error)})
            continue
        bodies[index] = body.instructions
        calls[index] = {
            ins.operand.rid
            for ins in body.instructions
            if ins.opcode.name in ("call", "callvirt", "ldftn", "ldvirtftn")
            and isinstance(ins.operand, Token)
            and ins.operand.table == 6
        }
    selected = {index for index, name in names.items() if re.search(pattern, name)}
    for _ in range(depth):
        selected |= {index for index, targets in calls.items() if targets & selected}
    result = {
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pattern": pattern,
        "caller_depth": depth,
        "errors": errors,
        "methods": [],
    }
    for index in sorted(selected):
        result["methods"].append(
            {
                "name": names[index],
                "token": hex(0x06000000 + index),
                "instructions": [
                    f"{ins.offset:04x} {ins.opcode.name} {operand_text(ins.operand)}"
                    for ins in bodies.get(index, [])
                ],
            }
        )
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print("Methods:", len(result["methods"]), "Parse errors:", len(errors))
    for method in result["methods"]:
        print(method["token"], method["name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--caller-depth", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.assembly, args.name, args.caller_depth, args.output)
