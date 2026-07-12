"""Table-extraction evaluation metrics: TEDS, number-fidelity, cell accuracy.

Scores a predicted table ({headers, rows}) against a hand-verified ground truth
of the same shape. Three complementary numbers because one lies:

  * TEDS         -- structure + content, robust to wrong row/col counts. The
                    OCRTurk / PubTabNet standard, so scores are comparable to
                    published numbers (PaddleOCR-VL TR TEDS ~0.87). Self-contained
                    Zhang-Shasha tree-edit distance (no apted dependency).
  * number_fid   -- did the FINANCIAL digits survive? Multiset recall of every
                    numeric cell's digit-string. A TEDS of 0.95 can still hide a
                    137,26 -> 1372,06 (10x) error; this isolates that risk.
  * cell_acc     -- positional exact-match ratio, only when shapes match. Human-
                    readable ("17/20 wrong") for debugging; N/A on shape mismatch.

Ground truth may set "exclude_cols" (e.g. a masked column no backend can read);
those columns are dropped from both pred and GT before scoring.
"""
import re

# Reuse the pipeline's digit extraction so eval and the production number-verify
# layer agree on what "the same number" means.
from pipeline.number_verify import _digits, is_numeric

_TR_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ç": "c", "Ç": "c", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u",
})


def fold(s) -> str:
    """Turkish-fold + lower + whitespace-collapse. Comparing folded strings keeps
    an OCR ı/i or ş/s slip from masquerading as a structural extraction error."""
    return re.sub(r"\s+", " ", str(s).translate(_TR_FOLD).lower()).strip()


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm_lev(a: str, b: str) -> float:
    """Normalized string edit distance in [0,1] -- the cell-content rename cost."""
    m = max(len(a), len(b))
    return 0.0 if m == 0 else _lev(a, b) / m


# ---------------------------------------------------------------------------
# TEDS: build each table as table->row->cell, compute Zhang-Shasha tree edit
# distance with a fractional rename cost on cell contents, normalize by node count.
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("tag", "text", "children")

    def __init__(self, tag, text="", children=None):
        self.tag = tag
        self.text = text
        self.children = children or []


def _build_tree(headers, rows, fold_text):
    def cell(text, tag):
        t = fold(text) if fold_text else str(text).strip()
        return _Node(tag, t)
    kids = []
    if headers:
        kids.append(_Node("tr", children=[cell(h, "th") for h in headers]))
    for row in rows:
        kids.append(_Node("tr", children=[cell(c, "td") for c in row]))
    return _Node("table", children=kids)


def _rename_cost(a: _Node, b: _Node) -> float:
    if a.tag != b.tag:
        return 1.0
    if a.tag in ("td", "th"):
        return _norm_lev(a.text, b.text)
    return 0.0


def _postorder(root):
    """Return (nodes, leftmost-leaf-descendant idx, keyroots) in 1-based postorder."""
    nodes, lmld = [], []

    def visit(n):
        first = None
        for c in n.children:
            idx = visit(c)
            if first is None:
                first = idx
        nodes.append(n)
        i = len(nodes)  # 1-based
        lmld.append(i if first is None else first)
        return i

    visit(root)
    last = {}
    for i in range(1, len(nodes) + 1):
        last[lmld[i - 1]] = i          # keyroot = largest-postorder node per lmld
    keyroots = sorted(last.values())
    return nodes, lmld, keyroots


def _tree_edit_distance(t1, t2):
    n1, l1, k1 = _postorder(t1)
    n2, l2, k2 = _postorder(t2)
    INF = float("inf")
    td = [[0.0] * (len(n2) + 1) for _ in range(len(n1) + 1)]

    def treedist(i, j):
        li, lj = l1[i - 1], l2[j - 1]
        fd = [[0.0] * (j - lj + 2) for _ in range(i - li + 2)]
        for di in range(1, i - li + 2):
            fd[di][0] = fd[di - 1][0] + 1            # delete
        for dj in range(1, j - lj + 2):
            fd[0][dj] = fd[0][dj - 1] + 1            # insert
        for di in range(1, i - li + 2):
            for dj in range(1, j - lj + 2):
                ni, nj = li + di - 1, lj + dj - 1
                if l1[ni - 1] == li and l2[nj - 1] == lj:
                    cost = _rename_cost(n1[ni - 1], n2[nj - 1])
                    fd[di][dj] = min(fd[di - 1][dj] + 1, fd[di][dj - 1] + 1,
                                     fd[di - 1][dj - 1] + cost)
                    td[ni][nj] = fd[di][dj]
                else:
                    pi, pj = l1[ni - 1] - li, l2[nj - 1] - lj
                    fd[di][dj] = min(fd[di - 1][dj] + 1, fd[di][dj - 1] + 1,
                                     fd[pi][pj] + td[ni][nj])
        return td

    for i in k1:
        for j in k2:
            treedist(i, j)
    return td[len(n1)][len(n2)], len(n1), len(n2)


def teds(pred, gt, fold_text=True) -> float:
    t1 = _build_tree(pred.get("headers", []), pred.get("rows", []), fold_text)
    t2 = _build_tree(gt.get("headers", []), gt.get("rows", []), fold_text)
    dist, n1, n2 = _tree_edit_distance(t1, t2)
    denom = max(n1, n2)
    return 1.0 if denom == 0 else round(1.0 - dist / denom, 4)


# ---------------------------------------------------------------------------
def _numeric_multiset(table):
    from collections import Counter
    c = Counter()
    for row in table.get("rows", []):
        for cell in row:
            if is_numeric(cell):
                d = _digits(cell)
                if d:
                    c[d] += 1
    return c


def number_fidelity(pred, gt) -> float:
    """Multiset recall of GT numeric cells' digit-strings in the prediction.
    Shape-independent: measures 'did the numbers survive', regardless of position
    (TEDS covers position). 1.0 when GT has no numbers."""
    g = _numeric_multiset(gt)
    p = _numeric_multiset(pred)
    total = sum(g.values())
    if total == 0:
        return 1.0
    matched = sum(min(cnt, p.get(k, 0)) for k, cnt in g.items())
    return round(matched / total, 4)


def cell_accuracy(pred, gt, fold_text=True):
    """Positional exact-match ratio over data cells -- only when shapes match,
    else None (use TEDS). Includes the header row."""
    ph, pr = pred.get("headers", []), pred.get("rows", [])
    gh, gr = gt.get("headers", []), gt.get("rows", [])
    if len(pr) != len(gr) or len(ph) != len(gh):
        return None
    if any(len(a) != len(b) for a, b in zip(pr, gr)):
        return None
    norm = fold if fold_text else (lambda x: str(x).strip())
    total = correct = 0
    for a, b in zip([ph] + pr, [gh] + gr):
        for x, y in zip(a, b):
            total += 1
            correct += (norm(x) == norm(y))
    return round(correct / total, 4) if total else None


def _drop_cols(table, cols):
    if not cols:
        return table
    cols = set(cols)
    keep = lambda seq: [v for i, v in enumerate(seq) if i not in cols]
    return {
        "headers": keep(table.get("headers", [])),
        "rows": [keep(r) for r in table.get("rows", [])],
    }


def score(pred, gt, fold_text=True) -> dict:
    """Full metric bundle for one predicted table vs its ground truth."""
    cols = gt.get("exclude_cols") or []
    p, g = _drop_cols(pred, cols), _drop_cols(gt, cols)
    return {
        "teds": teds(p, g, fold_text),
        "number_fid": number_fidelity(p, g),
        "cell_acc": cell_accuracy(p, g, fold_text),
        "shape_match": (len(p.get("rows", [])) == len(g.get("rows", []))
                        and len(p.get("headers", [])) == len(g.get("headers", []))),
        "pred_shape": (len(p.get("rows", [])), len(p.get("headers", []))),
        "gt_shape": (len(g.get("rows", [])), len(g.get("headers", []))),
    }
