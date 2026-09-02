"""仅依赖标准库的四图QC本地网页界面与可恢复审核记录。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Mapping, Sequence
from urllib.parse import unquote, urlparse

import pandas as pd

from .resources import sha256
from .schema import Participant


QC_KINDS = ("lesion_on_T1", "lesion_on_FLAIR", "WMH_lesion_overlap", "T1_macro20")
FAILURE_REASONS = (
    "t1_invalid",
    "flair_invalid",
    "registration_invalid",
    "wmh_failed",
    "macro_failed",
)


def review_database(qc_dir: Path) -> Path:
    return qc_dir / "qc_reviews.sqlite"


def review_tsv(derivatives: Path) -> Path:
    return derivatives / "tables" / "qc_reviews.tsv"


def qc_figure_paths(qc_dir: Path, participant_id: str) -> Dict[str, Path]:
    return {
        kind: qc_dir / "{}_{:02d}_{:s}.png".format(participant_id, index, kind)
        for index, kind in enumerate(QC_KINDS, start=1)
    }


def _combined_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii") if path.is_file() else b"missing")
        digest.update(b"\0")
    return digest.hexdigest()


def _status_hash(derivatives: Path, participant: Participant) -> str:
    status_dir = derivatives / participant.bids_id / "status"
    return _combined_hash([status_dir / "{}.json".format(stage) for stage in ("registration", "lesion", "wmh", "t1", "qc")])


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            participant_id TEXT PRIMARY KEY,
            review_state TEXT NOT NULL,
            qc_pass INTEGER NOT NULL DEFAULT 0,
            reasons_json TEXT NOT NULL DEFAULT '[]',
            note TEXT NOT NULL DEFAULT '',
            reviewer TEXT NOT NULL DEFAULT '',
            reviewed_at_utc TEXT NOT NULL DEFAULT '',
            image_hash TEXT NOT NULL,
            status_hash TEXT NOT NULL
        )
        """
    )
    return connection


def _sync_tsv(connection: sqlite3.Connection, participants: Sequence[Participant], output: Path) -> None:
    rows = []
    for participant in participants:
        record = connection.execute(
            "SELECT * FROM reviews WHERE participant_id=?", (participant.participant_id,)
        ).fetchone()
        if record is None:
            continue
        rows.append(
            {
                "participant_id": participant.participant_id,
                "review_state": record["review_state"],
                "qc_pass": bool(record["qc_pass"]),
                "failure_reasons": "|".join(json.loads(record["reasons_json"])),
                "note": record["note"],
                "reviewer": record["reviewer"],
                "reviewed_at_utc": record["reviewed_at_utc"],
                "image_hash": record["image_hash"],
                "processing_status_hash": record["status_hash"],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, sep="\t", index=False)
    temporary.replace(output)


def initialise_reviews(
    participants: Sequence[Participant], qc_dir: Path, derivatives: Path
) -> Path:
    """同步病例和图像指纹；图像或处理状态变化会使旧判定过期。"""

    database = review_database(qc_dir)
    with _connect(database) as connection:
        for participant in participants:
            figures = qc_figure_paths(qc_dir, participant.participant_id)
            image_hash = _combined_hash(list(figures.values()))
            status_hash = _status_hash(derivatives, participant)
            record = connection.execute(
                "SELECT review_state,image_hash,status_hash FROM reviews WHERE participant_id=?",
                (participant.participant_id,),
            ).fetchone()
            if record is None:
                connection.execute(
                    "INSERT INTO reviews(participant_id,review_state,image_hash,status_hash) VALUES(?,?,?,?)",
                    (participant.participant_id, "unreviewed", image_hash, status_hash),
                )
            else:
                changed = record["image_hash"] != image_hash or record["status_hash"] != status_hash
                state = "stale" if changed and record["review_state"] in {"pass", "fail"} else record["review_state"]
                connection.execute(
                    "UPDATE reviews SET review_state=?,image_hash=?,status_hash=? WHERE participant_id=?",
                    (state, image_hash, status_hash, participant.participant_id),
                )
        connection.commit()
        _sync_tsv(connection, participants, review_tsv(derivatives))
    return database


def load_review_table(participants: Sequence[Participant], qc_dir: Path, derivatives: Path) -> pd.DataFrame:
    """读取当前病例审核表；调用前同步指纹，禁止使用过期判定。"""

    database = initialise_reviews(participants, qc_dir, derivatives)
    with _connect(database) as connection:
        rows = [
            dict(
                connection.execute("SELECT * FROM reviews WHERE participant_id=?", (item.participant_id,)).fetchone()
            )
            for item in participants
        ]
    return pd.DataFrame(rows)


def _html() -> bytes:
    return r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WMH–T1 四图QC</title>
<style>
body{margin:0;background:#101418;color:#edf2f7;font-family:Arial,"Microsoft YaHei",sans-serif}header{position:sticky;top:0;z-index:2;background:#171d23;padding:10px 18px;display:flex;gap:14px;align-items:center;border-bottom:1px solid #34404b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:10px}.card{background:#1b2229;border:1px solid #34404b;border-radius:8px;overflow:hidden}.card h3{font-size:14px;margin:8px 12px}.card img{display:block;width:100%;height:38vh;object-fit:contain;background:#050607;cursor:zoom-in}.missing{height:38vh;display:flex;align-items:center;justify-content:center;color:#ffb4a2;background:#251b1b}.controls{padding:10px 18px 18px;display:grid;grid-template-columns:1fr auto;gap:12px}.reasons{display:flex;flex-wrap:wrap;gap:12px}.note{width:100%;min-height:52px;background:#0f1418;color:#fff;border:1px solid #44515d;border-radius:5px;padding:7px;box-sizing:border-box}button,select{background:#2b6cb0;color:#fff;border:0;border-radius:5px;padding:8px 12px}button.secondary{background:#46525d}button.danger{background:#a33b32}.status{color:#9fd4ff}.pass{color:#8fe3a5}.fail{color:#ffaaa2}
</style></head><body>
<header><strong>WMH–T1 四图QC</strong><button class="secondary" id="prev">上一例</button><button class="secondary" id="next">下一例</button><select id="jump"></select><span id="progress" class="status"></span><button id="finish">完成审核并关闭</button></header>
<main><div id="grid" class="grid"></div><div class="controls"><section><div class="reasons"><label><input type="checkbox" id="qc_pass"> QC通过</label><label><input type="checkbox" class="reason" value="t1_invalid"> T1不合格</label><label><input type="checkbox" class="reason" value="flair_invalid"> FLAIR不合格</label><label><input type="checkbox" class="reason" value="registration_invalid"> 配准不合格</label><label><input type="checkbox" class="reason" value="wmh_failed"> WMH失败</label><label><input type="checkbox" class="reason" value="macro_failed"> macro失败</label></div><p><textarea id="note" class="note" placeholder="备注（可选）"></textarea></p><span id="message"></span></section><div><button id="save">保存并到下一例</button></div></div></main>
<script>
let state={subjects:[],index:0}; const kinds=[['lesion_on_T1','lesion on T1'],['lesion_on_FLAIR','lesion on FLAIR'],['WMH_lesion_overlap','WMH–lesion overlap'],['T1_macro20','T1 macro20']];
const el=id=>document.getElementById(id); const reasons=()=>Array.from(document.querySelectorAll('.reason'));
function current(){return state.subjects[state.index]}
function render(){const s=current(); if(!s)return; el('jump').value=s.participant_id; el('progress').textContent=`${state.reviewed}/${state.total} 已完成 | 当前 ${state.index+1}/${state.total} | ${s.review_state}`; el('qc_pass').checked=s.qc_pass; reasons().forEach(x=>x.checked=s.reasons.includes(x.value)); el('note').value=s.note||''; const grid=el('grid'); grid.replaceChildren(); kinds.forEach(([kind,title])=>{const card=document.createElement('div');card.className='card';const h=document.createElement('h3');h.textContent=`${s.participant_id} — ${title}`;card.appendChild(h);if(s.images[kind]){const img=document.createElement('img');img.src=`/image/${encodeURIComponent(s.participant_id)}/${kind}?v=${s.image_hash}`;img.alt=title;img.onclick=()=>window.open(img.src,'_blank');card.appendChild(img)}else{const m=document.createElement('div');m.className='missing';m.textContent='该图未生成，请结合处理状态选择失败原因';card.appendChild(m)}grid.appendChild(card)});}
async function refresh(preferred){const response=await fetch('/api/state');state=await response.json();el('jump').replaceChildren(...state.subjects.map(s=>{const o=document.createElement('option');o.value=s.participant_id;o.textContent=`${s.participant_id} [${s.review_state}]`;return o}));let index=state.subjects.findIndex(s=>s.participant_id===preferred);if(index<0)index=state.resume_index;state.index=Math.max(0,index);render()}
el('qc_pass').onchange=()=>{if(el('qc_pass').checked)reasons().forEach(x=>x.checked=false)};reasons().forEach(x=>x.onchange=()=>{if(x.checked)el('qc_pass').checked=false});
el('prev').onclick=()=>{state.index=Math.max(0,state.index-1);render()};el('next').onclick=()=>{state.index=Math.min(state.total-1,state.index+1);render()};el('jump').onchange=()=>{state.index=state.subjects.findIndex(s=>s.participant_id===el('jump').value);render()};
el('save').onclick=async()=>{const s=current(), selected=reasons().filter(x=>x.checked).map(x=>x.value), pass=el('qc_pass').checked;if(!pass&&selected.length===0){el('message').textContent='请选择QC通过或至少一个失败原因';el('message').className='fail';return}const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({participant_id:s.participant_id,qc_pass:pass,reasons:selected,note:el('note').value})});const result=await response.json();if(!response.ok){el('message').textContent=result.error;return}await refresh(result.next_participant_id);el('message').textContent='已保存';el('message').className='pass'};
el('finish').onclick=async()=>{const response=await fetch('/api/finish',{method:'POST'});const result=await response.json();if(!response.ok){el('message').textContent=result.error;el('message').className='fail';return}document.body.innerHTML='<h2 style="padding:30px">全部审核已保存，可以关闭此页面。</h2>'};refresh();
</script></body></html>""".encode("utf-8")


class _QcServer(ThreadingHTTPServer):
    participants: Sequence[Participant]
    qc_dir: Path
    derivatives: Path


class _Handler(BaseHTTPRequestHandler):
    server: _QcServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: Mapping[str, object], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _records(self) -> List[Dict[str, object]]:
        table = load_review_table(self.server.participants, self.server.qc_dir, self.server.derivatives)
        rows: List[Dict[str, object]] = []
        for row in table.to_dict(orient="records"):
            paths = qc_figure_paths(self.server.qc_dir, str(row["participant_id"]))
            rows.append(
                {
                    "participant_id": row["participant_id"],
                    "review_state": row["review_state"],
                    "qc_pass": bool(row["qc_pass"]),
                    "reasons": json.loads(str(row["reasons_json"])),
                    "note": row["note"],
                    "image_hash": row["image_hash"],
                    "images": {kind: path.is_file() for kind, path in paths.items()},
                }
            )
        return rows

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = _html()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/state":
            subjects = self._records()
            reviewed = sum(row["review_state"] in {"pass", "fail"} for row in subjects)
            resume = next((index for index, row in enumerate(subjects) if row["review_state"] not in {"pass", "fail"}), 0)
            self._json({"subjects": subjects, "total": len(subjects), "reviewed": reviewed, "resume_index": resume})
            return
        components = parsed.path.strip("/").split("/")
        if len(components) == 3 and components[0] == "image":
            participant_id = unquote(components[1])
            kind = components[2]
            allowed = {item.participant_id for item in self.server.participants}
            if participant_id not in allowed or kind not in QC_KINDS:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = qc_figure_paths(self.server.qc_dir, participant_id)[kind]
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/review":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise ValueError("请求大小不合法")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                participant_id = str(payload["participant_id"])
                qc_pass = bool(payload.get("qc_pass", False))
                reasons = sorted(set(str(value) for value in payload.get("reasons", [])))
                if participant_id not in {item.participant_id for item in self.server.participants}:
                    raise ValueError("participant_id 不在当前审核清单")
                if any(value not in FAILURE_REASONS for value in reasons):
                    raise ValueError("包含未知失败原因")
                if qc_pass == bool(reasons):
                    raise ValueError("QC通过与失败原因必须且只能选择一类")
                note = str(payload.get("note", ""))[:4000]
                database = review_database(self.server.qc_dir)
                with _connect(database) as connection:
                    connection.execute(
                        """UPDATE reviews SET review_state=?,qc_pass=?,reasons_json=?,note=?,reviewer=?,reviewed_at_utc=?
                           WHERE participant_id=?""",
                        (
                            "pass" if qc_pass else "fail",
                            int(qc_pass),
                            json.dumps(reasons, ensure_ascii=False),
                            note,
                            os.environ.get("USER", os.environ.get("USERNAME", "")),
                            datetime.now(timezone.utc).isoformat(),
                            participant_id,
                        ),
                    )
                    connection.commit()
                    _sync_tsv(connection, self.server.participants, review_tsv(self.server.derivatives))
                records = self._records()
                next_id = next(
                    (str(row["participant_id"]) for row in records if row["review_state"] not in {"pass", "fail"}),
                    participant_id,
                )
                self._json({"status": "saved", "next_participant_id": next_id})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/finish":
            pending = [row["participant_id"] for row in self._records() if row["review_state"] not in {"pass", "fail"}]
            if pending:
                self._json({"error": "仍有{}例未完成审核".format(len(pending))}, HTTPStatus.CONFLICT)
                return
            self._json({"status": "complete"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def serve_qc(
    participants: Sequence[Participant], qc_dir: Path, derivatives: Path, port: int = 8765, open_browser: bool = True
) -> None:
    """在本机回环地址启动审核程序；关闭后所有结果仍保存在SQLite/TSV。"""

    initialise_reviews(participants, qc_dir, derivatives)
    server = _QcServer(("127.0.0.1", port), _Handler)
    server.participants = participants
    server.qc_dir = qc_dir
    server.derivatives = derivatives
    url = "http://127.0.0.1:{}/".format(port)
    print("QC GUI: {}".format(url), flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
