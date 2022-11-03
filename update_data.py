import requests
import MeanStars
import os

if __name__ == "__main__":
    fname = "EEM_dwarf_UBVIJHK_colors_Teff.txt"
    url = "http://www.pas.rochester.edu/~emamajek/" + fname
    res = requests.get(url)

    assert res.status_code == 200

    txt = res.content.decode().split("\n")

    linenos = []
    # look for lines starting with #SpT:
    for j, ll in enumerate(txt):
        if ll.startswith("#SpT"):
            linenos.append(j)

    # there should be only 2 of these
    assert len(linenos) == 2

    # generate copy of file with first header uncommented
    # and everything after second header commented
    outtxt = txt[: linenos[0]]
    outtxt.append(txt[linenos[0]].strip("#"))
    outtxt += txt[linenos[0] + 1 : linenos[1] + 1]
    for j in range(linenos[1] + 1, len(txt)):
        outtxt.append("#{}".format(txt[j]))

    outfile = os.path.join(os.path.split(MeanStars.__file__)[0], fname)
    with open(outfile, "w") as f:
        for ll in outtxt:
            f.write("{}\n".format(ll))
