def plog_to_chemkin(plog: dict) -> str:
    line = "{}\t{} {} {}\n".format(
        plog["name"], plog["parameters"][-1][1], plog["parameters"][-1][2], plog["parameters"][-1][3]
    )

    for i in range(0, len(plog["parameters"])):
        P, A, b, Ea = (
            plog["parameters"][i][0],
            plog["parameters"][i][1],
            plog["parameters"][i][2],
            plog["parameters"][i][3],
        )
        line += " PLOG / {:.5E}   {:.5E}  {:.5E}  {:.5E} /\n".format(P, A, b, Ea)

    if plog["is_duplicate"]:
        line += "DUP"

    return line


def arrheniusbase_to_chemkin(arrhenius: dict) -> str:
    line = "{}\t{:.5E}  {:.5E}  {:.5E}\n".format(
        arrhenius["name"], arrhenius["parameters"][0], arrhenius["parameters"][1], arrhenius["parameters"][2]
    )
    if arrhenius["is_duplicate"]:
        line += " DUPLICATE\n"

    return line


def comment_chemkin_string(chemkin_string: str) -> str:
    tmp = chemkin_string.split("\n")
    tmp = ["! " + i for i in tmp][:-1]
    commented_string = ""
    for i in tmp:
        commented_string += i + "\n"

    return commented_string
