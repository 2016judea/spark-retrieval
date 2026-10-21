"""Parse PubMed baseline XML.gz files into one parquet per file.

Usage: python scripts/parse_baseline.py [--workers 10]
Skips files whose parquet already exists. Only parses files listed OK in verify.log.
"""
import argparse, gzip, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa, pyarrow.parquet as pq
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/baseline"
OUT = ROOT / "data/parquet"

SCHEMA = pa.schema([
    ("pmid", pa.int64()), ("version", pa.int32()), ("title", pa.string()), ("abstract", pa.string()),
    ("journal", pa.string()), ("journal_abbr", pa.string()), ("year", pa.int32()),
    ("authors", pa.list_(pa.string())), ("mesh", pa.list_(pa.string())), ("mesh_major", pa.list_(pa.string())),
    ("chemicals", pa.list_(pa.string())), ("keywords", pa.list_(pa.string())),
    ("pub_types", pa.list_(pa.string())), ("doi", pa.string()), ("pmc", pa.string()),
    ("refs", pa.list_(pa.int64())), ("src_file", pa.int32()),
])


def text(el):
    return "".join(el.itertext()).strip() if el is not None else ""


def year_of(art):
    for path in ("Article/Journal/JournalIssue/PubDate/Year", "Article/ArticleDate/Year"):
        y = art.findtext(path)
        if y and y[:4].isdigit():
            return int(y[:4])
    md = art.findtext("Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
    for tok in md.split():
        if tok[:4].isdigit():
            return int(tok[:4])
    return None


def parse_file(name):
    out = OUT / name.replace(".xml.gz", ".parquet")
    if out.exists():
        return name, -1
    fileno = int(name[9:13])
    rows = {k: [] for k in SCHEMA.names}
    with gzip.open(SRC / name, "rb") as fh:
        for _, el in etree.iterparse(fh, tag="PubmedArticle", huge_tree=True):
            mc = el.find("MedlineCitation")
            pm = mc.find("PMID")
            art = mc.find("Article")
            abst = art.find("Abstract")
            parts = []
            if abst is not None:
                for at in abst.findall("AbstractText"):
                    lab = at.get("Label")
                    t = text(at)
                    parts.append(f"{lab}: {t}" if lab and lab.upper() != "UNLABELLED" else t)
            authors = []
            for a in art.findall("AuthorList/Author"):
                ln = a.findtext("LastName") or a.findtext("CollectiveName")
                if ln:
                    ini = a.findtext("Initials") or ""
                    authors.append(f"{ln} {ini}".strip())
            mesh, major = [], []
            for mh in mc.findall("MeshHeadingList/MeshHeading"):
                d = mh.find("DescriptorName")
                mesh.append(d.text)
                if d.get("MajorTopicYN") == "Y" or any(q.get("MajorTopicYN") == "Y" for q in mh.findall("QualifierName")):
                    major.append(d.text)
            ids = {i.get("IdType"): i.text for i in el.findall("PubmedData/ArticleIdList/ArticleId")}
            refs = []
            for r in el.findall("PubmedData/ReferenceList/Reference/ArticleIdList/ArticleId"):
                if r.get("IdType") == "pubmed" and r.text and r.text.strip().isdigit():
                    refs.append(int(r.text.strip()))
            rows["pmid"].append(int(pm.text)); rows["version"].append(int(pm.get("Version", "1")))
            rows["title"].append(text(art.find("ArticleTitle")) or text(art.find("VernacularTitle")))
            rows["abstract"].append(" ".join(parts))
            rows["journal"].append(art.findtext("Journal/Title") or "")
            rows["journal_abbr"].append(mc.findtext("MedlineJournalInfo/MedlineTA") or "")
            rows["year"].append(year_of(mc)); rows["authors"].append(authors)
            rows["mesh"].append(mesh); rows["mesh_major"].append(major)
            rows["chemicals"].append([c.text for c in mc.findall("ChemicalList/Chemical/NameOfSubstance")])
            rows["keywords"].append([text(k) for k in mc.findall("KeywordList/Keyword")])
            rows["pub_types"].append([p.text for p in art.findall("PublicationTypeList/PublicationType")])
            rows["doi"].append(ids.get("doi")); rows["pmc"].append(ids.get("pmc"))
            rows["refs"].append(refs); rows["src_file"].append(fileno)
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.table(rows, schema=SCHEMA), tmp, compression="zstd")
    tmp.rename(out)
    return name, len(rows["pmid"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ok = sorted(l.split()[1] for l in (SRC / "verify.log").read_text().splitlines() if l.startswith("OK "))
    if a.limit:
        ok = ok[: a.limit]
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for name, c in ex.map(parse_file, ok):
            if c > 0:
                n += c
    print(f"parsed {len(ok)} files, {n} new rows, {time.time()-t0:.0f}s", flush=True)
